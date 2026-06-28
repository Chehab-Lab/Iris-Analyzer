"""
UI logic for the Iris Analyzer app.

Owns everything the user interacts with: the nav bar, the input controls
(eye-side dropdown, Upload image button, camera), and result rendering. All
heavy lifting is delegated to ``analysis``; this module holds no image or iris
processing of its own.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

import analysis
from analysis import IrisResult

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

CHEHAB_LAB_URL = "https://chehablab.com"

# Custom component: capture a webcam frame at the camera's highest resolution
# (st.camera_input only captures the browser's default stream size).
_camera_hires = components.declare_component(
    "camera_hires",
    path=str(Path(__file__).parent / "components" / "camera_hires"),
)


def hires_camera(key: str = "camera") -> Optional[bytes]:
    """Render the hi-res webcam component; return JPEG bytes once captured."""
    data_url = _camera_hires(key=key, default=None)
    if not data_url or "," not in data_url:
        return None
    try:
        return base64.b64decode(data_url.split(",", 1)[1])
    except Exception:
        return None


@lru_cache(maxsize=2)
def _asset_data_uri(filename: str) -> str:
    """Return an asset image as an inline base64 data URI (cached)."""
    try:
        path = Path(__file__).parent / "assets" / filename
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _logo_data_uri() -> str:
    return _asset_data_uri("chehab-lab-logo.png")


def render_navbar(page: str, *, work_title: str = "") -> None:
    """Fixed top bar with the app title and page links."""
    home_cls = "active" if page != "about" and not work_title else ""
    about_cls = "active" if page == "about" else ""
    if work_title:
        st.markdown(
            f"""
            <div class="iris-nav">
                <a class="nav-back" href="?reset=1" target="_self">←</a>
                <span class="name work-name">{work_title}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <div class="iris-nav">
            <span class="glyph">\U0001F441</span>
            <div class="nav-brand">
                <span class="name">Pupillometry</span>
                <span class="tagline">Iris-to-Pupil ratio</span>
            </div>
            <div class="nav-links">
                <a class="{home_cls}" href="?" target="_self">Analyzer</a>
                <a class="{about_cls}" href="?page=about" target="_self">About</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def _active_adj_keys() -> tuple[str, str, str]:
    if st.session_state.get("adj_target", "pupil") == "pupil":
        return "adj_pcx", "adj_pcy", "adj_pr"
    return "adj_icx", "adj_icy", "adj_ir"



def _move_center(dx: float, dy: float, w: float, h: float) -> None:
    cx_k, cy_k, _ = _active_adj_keys()
    st.session_state[cx_k] = _clamp(st.session_state.get(cx_k, 0.0) + dx, 0.0, w)
    st.session_state[cy_k] = _clamp(st.session_state.get(cy_k, 0.0) + dy, 0.0, h)


def _adjust_radius(delta: float, rmax: float) -> None:
    _, _, r_k = _active_adj_keys()
    st.session_state[r_k] = _clamp(st.session_state.get(r_k, 1.0) + delta, 1.0, rmax)


_RADIUS_STEP = 1.0

# Inline SVG icons — four arrows in/out for radius controls.
_FINE_ICON_OUT = """
<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
  <polygon points="12,2 10,6 14,6" fill="currentColor"/>
  <polygon points="12,22 10,18 14,18" fill="currentColor"/>
  <polygon points="2,12 6,10 6,14" fill="currentColor"/>
  <polygon points="22,12 18,10 18,14" fill="currentColor"/>
</svg>
""".strip()

_FINE_ICON_IN = """
<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
  <polygon points="12,8 10,4 14,4" fill="currentColor"/>
  <polygon points="12,16 10,20 14,20" fill="currentColor"/>
  <polygon points="8,12 4,10 4,14" fill="currentColor"/>
  <polygon points="16,12 20,10 20,14" fill="currentColor"/>
</svg>
""".strip()


def _set_pupil_target() -> None:
    st.session_state["adj_target"] = "pupil"


def _set_iris_target() -> None:
    st.session_state["adj_target"] = "iris"


def _render_correction_adj_hooks(w: float, h: float, rmax: float) -> None:
    """Hidden Streamlit buttons — the HTML pad bridge clicks these (no page reload)."""
    st.markdown('<div id="correction-adj-hooks"></div>', unsafe_allow_html=True)
    st.button("▲", key="adj_mv_up", on_click=_move_center, args=(0.0, -2.0, w, h))
    st.button("◀", key="adj_mv_left", on_click=_move_center, args=(-2.0, 0.0, w, h))
    st.button("▶", key="adj_mv_right", on_click=_move_center, args=(2.0, 0.0, w, h))
    st.button("▼", key="adj_mv_down", on_click=_move_center, args=(0.0, 2.0, w, h))
    st.button("out", key="adj_fine_out", on_click=_adjust_radius, args=(_RADIUS_STEP, rmax))
    st.button("in", key="adj_fine_in", on_click=_adjust_radius, args=(-_RADIUS_STEP, rmax))
    st.button("Pupil", key="adj_sel_pupil", on_click=_set_pupil_target)
    st.button("Iris", key="adj_sel_iris", on_click=_set_iris_target)


def _render_correction_bridge() -> None:
    """Wire HTML pad taps to hidden Streamlit buttons without reloading the page."""
    st.html(
        """
        <script>
        (function () {
            if (window.__irisAdjBridge) return;
            window.__irisAdjBridge = true;
            function clickAdj(key) {
                const root = document.querySelector('[class*="st-key-' + key + '"]');
                const btn = root && root.querySelector("button");
                if (btn) btn.click();
            }
            document.addEventListener("click", function (e) {
                const t = e.target.closest("#correction-ui a[data-adj]");
                if (!t) return;
                e.preventDefault();
                clickAdj(t.getAttribute("data-adj"));
            }, true);
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def _render_android_correction(w: int, h: int, rmax: float) -> None:
    """Pupil/Iris selector + arrow pads — HTML design with hidden Streamlit hooks."""
    wf, hf = float(w), float(h)
    st.session_state.setdefault("adj_target", "pupil")
    target = st.session_state["adj_target"]
    pup_cls = "sel-btn active" if target == "pupil" else "sel-btn"
    ir_cls = "sel-btn active" if target == "iris" else "sel-btn"

    st.markdown(
        f"""
        <div id="correction-ui" class="manual-correction-panel compact">
            <h4>Manual Correction</h4>
            <div class="correction-select">
                <a href="#" class="{pup_cls}" data-adj="adj_sel_pupil">Pupil</a>
                <a href="#" class="{ir_cls}" data-adj="adj_sel_iris">Iris</a>
            </div>
            <div class="correction-pads-row">
                <div class="correction-pad">
                    <div class="pad-label">Move center</div>
                    <div class="dpad-grid">
                        <span class="dpad-gap"></span>
                        <a href="#" class="pad-btn" data-adj="adj_mv_up">▲</a>
                        <span class="dpad-gap"></span>
                        <a href="#" class="pad-btn" data-adj="adj_mv_left">◀</a>
                        <span class="dpad-gap"></span>
                        <a href="#" class="pad-btn" data-adj="adj_mv_right">▶</a>
                        <span class="dpad-gap"></span>
                        <a href="#" class="pad-btn" data-adj="adj_mv_down">▼</a>
                        <span class="dpad-gap"></span>
                    </div>
                </div>
                <div class="correction-pad fine-pad">
                    <div class="pad-label">Adjust size</div>
                    <div class="fine-btns">
                        <a href="#" class="pad-btn fine-btn" data-adj="adj_fine_out"
                           title="Enlarge selected circle">{_FINE_ICON_OUT}</a>
                        <a href="#" class="pad-btn fine-btn" data-adj="adj_fine_in"
                           title="Shrink selected circle">{_FINE_ICON_IN}</a>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_correction_adj_hooks(wf, hf, rmax)
    _render_correction_bridge()


def _render_framed_image(image_bytes: bytes, *, sticky: bool = False) -> None:
    """Show an image inside the gray preview panel."""
    mime = "image/jpeg" if image_bytes[:3] == b"\xff\xd8\xff" else "image/png"
    b64 = base64.b64encode(image_bytes).decode("ascii")
    sticky_cls = " preview-sticky" if sticky else ""
    st.markdown(
        f'<div class="preview-frame{sticky_cls}">'
        f'<img src="data:{mime};base64,{b64}" alt="Preview" />'
        f"</div>",
        unsafe_allow_html=True,
    )


def _center_crop_bytes(image_bytes: bytes, margin_pct: int) -> bytes:
    """UI-only crop helper — trims edges before processing (like Android UCrop)."""
    if Image is None or margin_pct <= 0:
        return image_bytes
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    iw, ih = img.size
    mx = int(iw * margin_pct / 200)
    my = int(ih * margin_pct / 200)
    cropped = img.crop((mx, my, iw - mx, ih - my))
    out = io.BytesIO()
    fmt = "JPEG" if image_bytes[:3] == b"\xff\xd8\xff" else "PNG"
    cropped.save(out, format=fmt, quality=92)
    return out.getvalue()


def _go_home() -> None:
    """Return to the upload/home screen and clear the in-progress image session."""
    for key in (
        "pending_bytes", "pending_name", "pending_eye_side", "pending_model",
        "img_sig", "analysis_cache", "workflow", "adj_sig", "captured_image",
    ):
        st.session_state.pop(key, None)
    for key in ("view", "adj", "reset", "cam"):
        if key in st.query_params:
            del st.query_params[key]
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


def _go_workflow(step: str) -> None:
    st.session_state["workflow"] = step
    view_map = {
        "camera": "camera",
        "preview": "preview",
        "crop": "crop",
        "result": "result",
        "correction": "correction",
    }
    if step in view_map:
        st.query_params["view"] = view_map[step]
    elif "view" in st.query_params:
        del st.query_params["view"]


def _sync_workflow_from_url() -> None:
    view = st.query_params.get("view")
    if view == "camera":
        st.session_state["workflow"] = "camera"
        return
    mapping = {
        "preview": "preview",
        "crop": "crop",
        "result": "result",
        "correction": "correction",
    }
    if view in mapping and st.session_state.get("pending_bytes"):
        st.session_state["workflow"] = mapping[view]


def _work_titles() -> dict[str, str]:
    return {
        "camera": "Take Photo",
        "preview": "Preview",
        "crop": "Adjust Size",
        "result": "Results",
        "correction": "Manual Correction",
    }


def _render_camera_step(eye_side: str, model: str) -> None:
    """Dedicated full-page camera capture — rear camera on phones, webcam on laptops."""
    st.markdown('<div id="camera-screen-anchor"></div>', unsafe_allow_html=True)
    st.caption(
        "Fill the frame with the eye. On phones the **rear camera** opens by default — "
        "use **Flip camera** if needed."
    )
    shot = hires_camera(key=f"camera_{st.session_state.get('camera_key', 0)}")
    if shot is not None:
        st.session_state.pending_bytes = shot
        st.session_state.pending_name = "captured.jpg"
        st.session_state.pending_eye_side = eye_side
        st.session_state.pending_model = model
        st.session_state.img_sig = (hash(shot), eye_side, model)
        st.session_state.pop("analysis_cache", None)
        st.session_state.pop("captured_image", None)
        _go_workflow("preview")
        st.rerun()

    if st.button("Cancel", use_container_width=True):
        _go_home()
        st.rerun()


def _render_preview_step(image_bytes: bytes) -> None:
    """Step 1 — preview captured/uploaded image before processing."""
    _render_framed_image(image_bytes)

    if st.button("Adjust Image Size", use_container_width=True):
        _go_workflow("crop")
        st.rerun()
    if st.button("Start Processing", use_container_width=True, type="primary"):
        _go_workflow("result")
        st.session_state.pop("analysis_cache", None)
        st.rerun()


def _render_crop_step(image_bytes: bytes, eye_side: str, model: str) -> None:
    """Step 1b — trim edges before returning to preview."""
    st.markdown("**Adjust Image Size**")
    margin = st.slider("Crop amount", 0, 40, st.session_state.get("crop_margin", 0))
    st.session_state["crop_margin"] = margin
    if margin > 20:
        st.warning("Heavy cropping can cut off the iris or pupil and cause analysis to fail.")
    preview = _center_crop_bytes(image_bytes, margin)
    _render_framed_image(preview)

    if st.button("Apply crop", use_container_width=True, type="primary"):
        st.session_state.pending_bytes = preview
        st.session_state.img_sig = (hash(preview), eye_side, model)
        st.session_state.pop("analysis_cache", None)
        _go_workflow("preview")
        st.rerun()
    if st.button("Cancel", use_container_width=True):
        _go_workflow("preview")
        st.rerun()


def render_result(r: IrisResult, eye_side: str, *, correction: bool = False) -> None:
    """Show the (adjustable) overlay, nudge controls, metrics and downloads."""
    if not r.ok:
        st.error(r.error or "Analysis failed.")
        if r.original_png:
            st.image(r.original_png, use_container_width=True, caption="Source image")
        return

    if not r.original_png:
        st.image(r.overlay_png, use_container_width=True,
                 caption="Pupil & iris boundaries")
        return

    w, h = max(1, int(r.width)), max(1, int(r.height))
    rmax = float(max(w, h))

    # Seed the adjustable values to the detected ones whenever the result changes.
    sig = f"{w}x{h}|{r.pupil_center}|{r.pupil_radius}|{r.iris_center}|{r.iris_radius}"
    if st.session_state.get("adj_sig") != sig:
        st.session_state["adj_sig"] = sig
        st.session_state["adj_pcx"] = _clamp(r.pupil_center[0], 0.0, w)
        st.session_state["adj_pcy"] = _clamp(r.pupil_center[1], 0.0, h)
        st.session_state["adj_pr"] = _clamp(r.pupil_radius, 1.0, rmax)
        st.session_state["adj_icx"] = _clamp(r.iris_center[0], 0.0, w)
        st.session_state["adj_icy"] = _clamp(r.iris_center[1], 0.0, h)
        st.session_state["adj_ir"] = _clamp(r.iris_radius, 1.0, rmax)

    # nudge callbacks have already run, so these are the current values
    pcx, pcy, pr = st.session_state["adj_pcx"], st.session_state["adj_pcy"], st.session_state["adj_pr"]
    icx, icy, ir = st.session_state["adj_icx"], st.session_state["adj_icy"], st.session_state["adj_ir"]
    ipr = ir / pr if pr > 0 else 0.0

    # Fast, constant-size overlay (OpenCV). st.image sends a small /media URL
    # reference — not a large inline message — so rapid nudging won't thrash
    # Streamlit's ForwardMsg cache, and constant dimensions keep the layout still.
    overlay = analysis.draw_circles_png(r.original_png, (pcx, pcy), pr, (icx, icy), ir)

    screen_cls = "work-screen correction-mode" if correction else "work-screen"
    st.markdown(f'<div class="{screen_cls}">', unsafe_allow_html=True)
    _render_framed_image(overlay, sticky=correction)
    ipr_cls = "ocu-ipr compact" if correction else "ocu-ipr"
    st.markdown(
        f'<div class="{ipr_cls}">IPR: {ipr:.3f}</div>',
        unsafe_allow_html=True,
    )

    if correction:
        _render_android_correction(w, h, rmax)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Correct Ratio", use_container_width=True):
            st.session_state["adj_sig"] = None
            st.rerun()
        if st.button("Done", use_container_width=True, type="primary"):
            _go_workflow("result")
            st.rerun()
    else:
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Correct Ratio", use_container_width=True):
            _go_workflow("correction")
            st.rerun()
        if st.button("Retry", use_container_width=True):
            _go_workflow("preview")
            st.session_state.pop("analysis_cache", None)
            st.session_state.pop("captured_image", None)
            st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
            st.rerun()

        with st.expander("Measurement details", expanded=False):
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-row">
                        <div class="metric hl"><div class="label">IPR</div>
                            <div class="value">{ipr:.4f}</div></div>
                        <div class="metric"><div class="label">Eye side</div>
                            <div class="value text">{eye_side.title()}</div></div>
                        <div class="metric"><div class="label">Iris radius</div>
                            <div class="value">{ir:.1f}<span class="unit"> px</span></div></div>
                        <div class="metric"><div class="label">Pupil radius</div>
                            <div class="value">{pr:.1f}<span class="unit"> px</span></div></div>
                    </div>
                    <div class="metric-row">
                        <div class="metric"><div class="label">Iris center</div>
                            <div class="value coords">{icx:.0f}, {icy:.0f}</div></div>
                        <div class="metric"><div class="label">Pupil center</div>
                            <div class="value coords">{pcx:.0f}, {pcy:.0f}</div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv = (
            "image,eye_side,ipr,pupil_x,pupil_y,pupil_radius,"
            "iris_x,iris_y,iris_radius,width,height\n"
            f"{r.name},{eye_side},{ipr:.5f},{pcx:.3f},{pcy:.3f},{pr:.3f},"
            f"{icx:.3f},{icy:.3f},{ir:.3f},{w},{h}\n"
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download overlay (PNG)",
                data=overlay,
                file_name=f"iris_overlay_{stamp}.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Download measurements (CSV)",
                data=csv.encode("utf-8"),
                file_name=f"iris_result_{stamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )


_MODELS = {
    "Open-iris": "open-iris",
    "Open-iris + MediaPipe": "combined",
}


def _collect_image(*, show_controls: bool = True) -> tuple[Optional[bytes], str, str, str]:
    """Render upload/camera controls and return (image_bytes, name, eye_side, model)."""
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("camera_key", 0)

    if not show_controls:
        eye_side = st.session_state.get("pending_eye_side", "right")
        model = st.session_state.get("pending_model", "open-iris")
        return None, "", eye_side, model

    eye_col, mdl_col, up_col, cam_col = st.columns([1, 1.5, 1, 1])
    with eye_col:
        eye_side = st.selectbox("Eye side", ["right", "left"], index=0)
    with mdl_col:
        model = _MODELS[st.selectbox("Model", list(_MODELS.keys()), index=0)]
    with up_col:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        up = st.file_uploader(
            "Upload an iris image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state['uploader_key']}",
        )
    with cam_col:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if st.button("Take an image", use_container_width=True, type="primary"):
            st.session_state.pending_eye_side = eye_side
            st.session_state.pending_model = model
            st.session_state.pop("pending_bytes", None)
            st.session_state.pop("captured_image", None)
            st.session_state["camera_key"] += 1
            st.session_state["uploader_key"] += 1
            _go_workflow("camera")
            st.rerun()

    image_bytes: Optional[bytes] = None
    image_name = "captured.png"

    if up is not None:
        image_bytes = up.getvalue()
        image_name = up.name
        st.session_state.pop("captured_image", None)

    return image_bytes, image_name, eye_side, model


def render_footer() -> None:
    """Bottom credit line; only the Chehab Lab logo links to the site."""
    logo = _logo_data_uri()
    badge = (f'<img src="{logo}" alt="Chehab Lab" />' if logo
             else '<span class="lab">Chehab Lab</span>')
    st.markdown(
        f'<div class="iris-footer">Made with <span class="heart">❤</span> by '
        f'<a href="{CHEHAB_LAB_URL}" target="_blank" rel="noopener noreferrer">'
        f'{badge}</a> @ 2026</div>',
        unsafe_allow_html=True,
    )


def _render_workflow_page(
    pending: bytes,
    pname: str,
    eye_side: str,
    model: str,
    workflow: str,
) -> None:
    """Dedicated screen for preview / crop / results — no upload controls."""
    if workflow == "preview":
        _render_preview_step(pending)
        return
    if workflow == "crop":
        _render_crop_step(pending, eye_side, model)
        return

    cache_key = (hash(pending), eye_side, model)
    cached = st.session_state.get("analysis_cache")
    if not cached or cached.get("key") != cache_key:
        analyzing = st.empty()
        analyzing.markdown(
            '<div class="iris-analyzing">Analyzing…</div>',
            unsafe_allow_html=True,
        )
        result = analysis.analyze_one(pending, pname, eye_side, model)
        st.session_state["analysis_cache"] = {"key": cache_key, "result": result}
        analyzing.empty()
    else:
        result = cached["result"]

    render_result(result, eye_side, correction=(workflow == "correction"))


def render_analyzer() -> None:
    """Home upload screen, or a focused workflow page once an image is in progress."""
    if st.query_params.get("reset") == "1":
        _go_home()
        st.rerun()

    _sync_workflow_from_url()
    workflow = st.session_state.get("workflow")
    pending = st.session_state.get("pending_bytes")

    if workflow == "camera":
        eye_side = st.session_state.get("pending_eye_side", "right")
        model = st.session_state.get("pending_model", "open-iris")
        if not analysis.model_ready(model):
            st.error("The selected model is not available in this environment.")
            return
        _render_camera_step(eye_side, model)
        return

    in_workflow = pending and workflow in ("preview", "crop", "result", "correction")

    if in_workflow:
        eye_side = st.session_state.get("pending_eye_side", "right")
        model = st.session_state.get("pending_model", "open-iris")
        pname = st.session_state.get("pending_name", "image.png")
        if not analysis.model_ready(model):
            st.error("The selected model is not available in this environment.")
            return
        _render_workflow_page(pending, pname, eye_side, model, workflow)
        return

    image_bytes, image_name, eye_side, model = _collect_image(show_controls=True)

    if not analysis.model_ready(model):
        missing = []
        if not analysis.CV2_OK:
            missing.append("opencv")
        if not analysis.IRIS_OK:
            missing.append("open-iris engine")
        if model == "combined" and not analysis.MP_OK:
            missing.append("mediapipe")
        st.caption("The selected model is not available in this environment "
                   f"(missing: {', '.join(missing) or 'unknown'}).")
        if model == "combined" and not analysis.MP_OK and analysis.MP_ERR:
            st.code(analysis.MP_ERR, language="text")
        return

    if image_bytes:
        sig = (hash(image_bytes), eye_side, model)
        if st.session_state.get("img_sig") != sig:
            st.session_state.img_sig = sig
            st.session_state.pending_bytes = image_bytes
            st.session_state.pending_name = image_name
            st.session_state.pending_eye_side = eye_side
            st.session_state.pending_model = model
            st.session_state.pop("analysis_cache", None)
            _go_workflow("preview")
            st.rerun()


def render_about() -> None:
    """Static About page: what the app does, the output, authors and license."""
    st.markdown(
        """
        <div class="iris-card">
            <h3>What is Pupillometry?</h3>
            <p>Pupillometry is a tool for measuring the geometry of the human eye
            from iris images — including near-infrared (IR) and standard RGB photos.
            For a single image it removes specular reflections inside the pupil, runs the <b>open-iris</b>
            recognition pipeline to segment the eye, estimates the pupil and iris
            circles, and reports the <b>Iris-to-Pupil Ratio (IPR)</b> — a normalized
            measure of pupil dilation that is independent of image scale.</p>
            <p>Pick the eye side, provide an image, and the app returns an annotated
            overlay together with the numeric measurements, both downloadable.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    output = _asset_data_uri("output.png")
    output_img = (f'<img src="{output}" alt="Example output overlay" />'
                  if output else "")
    st.markdown(
        f"""
        <div class="iris-card">
            <h3>Understanding the output</h3>
            <div class="about-output">
                <div class="shot">{output_img}</div>
                <div class="legend">
                    <p>Every analyzed image produces an annotated overlay marking the
                    detected geometry:</p>
                    <ul>
                        <li><span class="key" style="color:#16a34a">●</span>
                            <b>Pupil circle</b> — the fitted pupil boundary</li>
                        <li><span class="key" style="color:#2563eb">●</span>
                            <b>Iris boundary</b> — the outer iris limbus</li>
                        <li><span class="key" style="color:#7c3aed">●</span>
                            <b>Iris center</b></li>
                        <li><span class="key" style="color:#dc2626">●</span>
                            <b>Pupil center</b></li>
                    </ul>
                    <p>From these the app computes the <b>IPR = iris radius / pupil
                    radius</b>, plus the pupil and iris radii and centers (in pixels),
                    which you can export as a CSV.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="iris-card">
            <h3>Models</h3>
            <p><b>Open-iris</b> — a deep-learning segmentation pipeline (open-iris)
            trained on <b>near-infrared, close-up</b> iris images. It detects both the
            iris and the pupil. Most accurate on uploaded IR iris photos or images
            from a dedicated/medical iris camera. In ordinary visible light or from a
            distance it can struggle and may confuse the iris with the pupil.</p>
            <p><b>Open-iris + MediaPipe</b> — the <b>iris</b> boundary comes from
            MediaPipe FaceMesh (robust on visible-light webcam frames where the whole
            eye/face is visible) and the <b>pupil</b> from open-iris. Best for ordinary
            webcam captures. It needs a visible face/eye, so it won't work on a tight
            iris close-up with no surrounding face.</p>
            <p><b>Rule of thumb:</b> average webcam → <b>Open-iris + MediaPipe</b>;
            uploaded images or a medical/IR iris camera → <b>Open-iris</b>.</p>
        </div>

        <div class="iris-card">
            <h3>Authors</h3>
            <p>Noha Faour &nbsp;·&nbsp; Ahmad Mustapha &nbsp;·&nbsp; Ali Chehab</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="iris-card">
            <h3>License</h3>
            <p>© 2026 Chehab Lab. All rights reserved.</p>
            <p>This work is licensed under
            <a href="https://creativecommons.org/licenses/by-nc/4.0/"
               target="_blank" rel="noopener noreferrer">Creative Commons
            Attribution–NonCommercial 4.0 (CC BY-NC 4.0)</a> — you may use and share
            it with attribution, for non-commercial purposes only.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run() -> None:
    """Route between the Analyzer and About pages; always show the footer."""
    page = st.query_params.get("page", "analyzer")
    work_title = ""
    if page != "about":
        _sync_workflow_from_url()
        wf = st.session_state.get("workflow")
        if wf in _work_titles() and (wf == "camera" or st.session_state.get("pending_bytes")):
            work_title = _work_titles()[wf]
    render_navbar(page, work_title=work_title)

    if page == "about":
        render_about()
    else:
        render_analyzer()

    render_footer()
