"""
UI logic for the Iris Analyzer app.

Owns everything the user interacts with: the nav bar, the input controls
(eye-side dropdown, Upload image button, camera), and result rendering. All
heavy lifting is delegated to ``analysis``; this module holds no image or iris
processing of its own.
"""

from __future__ import annotations

import base64
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

import analysis
from analysis import IrisResult

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


def render_navbar(page: str) -> None:
    """Fixed top bar with the app title and page links."""
    home_cls = "active" if page != "about" else ""
    about_cls = "active" if page == "about" else ""
    st.markdown(
        f"""
        <div class="iris-nav">
            <span class="glyph">\U0001F441</span>
            <span class="name">Iris Analyzer</span>
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


def _bump(key: str, delta: float, lo: float, hi: float) -> None:
    """on_click callback — runs before the rerun, so the new value is rendered."""
    st.session_state[key] = _clamp(st.session_state.get(key, lo) + delta, lo, hi)


def _nudge(label: str, key: str, lo: float, hi: float, step: float = 1.0) -> None:
    """A label with −/+ buttons that nudge a session_state value within [lo, hi]."""
    c0, c1, c2, c3 = st.columns([3, 1.1, 1.1, 1.1], vertical_alignment="center")
    c0.markdown(f"<div style='font-size:.85rem'>{label}</div>", unsafe_allow_html=True)
    c1.button("−", key=f"{key}_dec", use_container_width=True,
              on_click=_bump, args=(key, -step, lo, hi))
    c2.markdown(
        f"<div style='text-align:center;font-weight:600;font-size:.9rem'>"
        f"{st.session_state[key]:.0f}</div>",
        unsafe_allow_html=True,
    )
    c3.button("+", key=f"{key}_inc", use_container_width=True,
              on_click=_bump, args=(key, step, lo, hi))


def render_result(r: IrisResult, eye_side: str) -> None:
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

    img_col, data_col = st.columns([3, 2], gap="large")
    with img_col:
        st.image(overlay, use_container_width=True, caption="Pupil & iris boundaries")
        st.caption("Nudge the centers and radii (± 1 px) to refine the boundaries.")
        pcol, icol = st.columns(2, gap="large")
        with pcol:
            st.markdown("<b style='color:#16a34a'>Pupil</b>", unsafe_allow_html=True)
            _nudge("Center X", "adj_pcx", 0.0, float(w))
            _nudge("Center Y", "adj_pcy", 0.0, float(h))
            _nudge("Radius", "adj_pr", 1.0, rmax)
        with icol:
            st.markdown("<b style='color:#2563eb'>Iris</b>", unsafe_allow_html=True)
            _nudge("Center X", "adj_icx", 0.0, float(w))
            _nudge("Center Y", "adj_icy", 0.0, float(h))
            _nudge("Radius", "adj_ir", 1.0, rmax)
    with data_col:
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric hl"><div class="label">IPR</div>
                    <div class="value">{ipr:.4f}</div></div>
                <div class="metric"><div class="label">Eye side</div>
                    <div class="value" style="font-size:1.1rem">{eye_side.title()}</div></div>
                <div class="metric"><div class="label">Iris radius</div>
                    <div class="value">{ir:.1f}<span class="unit"> px</span></div></div>
                <div class="metric"><div class="label">Pupil radius</div>
                    <div class="value">{pr:.1f}<span class="unit"> px</span></div></div>
                <div class="metric"><div class="label">Iris center</div>
                    <div class="value" style="font-size:1rem">{icx:.0f}, {icy:.0f}</div></div>
                <div class="metric"><div class="label">Pupil center</div>
                    <div class="value" style="font-size:1rem">{pcx:.0f}, {pcy:.0f}</div></div>
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
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
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


def _collect_image() -> tuple[Optional[bytes], str, str, str]:
    """Render the controls and return (image_bytes, image_name, eye_side, model).

    The camera's open/closed state lives in the URL (?cam=1) rather than in
    session_state: granting camera permission requires a page reload, which
    wipes session_state — but the URL survives, so the camera stays visible and
    works after the reload instead of disappearing.
    """
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("camera_key", 0)
    cam_on = st.query_params.get("cam") == "1"

    eye_col, mdl_col, up_col, cam_col = st.columns([1, 1.5, 1, 1])
    with eye_col:
        eye_side = st.selectbox("Eye side", ["right", "left"], index=0)
    with mdl_col:
        model = _MODELS[st.selectbox("Model", list(_MODELS.keys()), index=0)]
    with up_col:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        up = st.file_uploader(
            "Upload an IR iris image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state['uploader_key']}",
        )
    with cam_col:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        label = "Close camera" if cam_on else "Take an image"
        if st.button(label, use_container_width=True, type="primary"):
            if cam_on:
                del st.query_params["cam"]
            else:
                # open a fresh camera: drop any previous photo / uploaded image
                st.query_params["cam"] = "1"
                st.session_state["camera_key"] += 1
                st.session_state["uploader_key"] += 1
                st.session_state.pop("captured_image", None)
            st.rerun()

    image_bytes: Optional[bytes] = None
    image_name = "captured.png"

    if up is not None:
        image_bytes = up.getvalue()
        image_name = up.name
        # uploading an image drops any captured photo and closes the camera
        st.session_state.pop("captured_image", None)
        if cam_on:
            del st.query_params["cam"]
            st.rerun()

    if cam_on:
        shot = hires_camera(key=f"camera_{st.session_state['camera_key']}")
        st.caption("The iris detection engine expects in-distribution, high-resolution "
                   "close-up images of an eye. Use the web-camera results with caution.")
        if shot is not None:
            # photo taken: keep it, stop the live stream, then show the analysis
            st.session_state["captured_image"] = shot
            del st.query_params["cam"]
            st.rerun()
        else:
            st.caption("After clicking **Allow** for camera access, reload the page — "
                       "the camera will stay open and start working.")

    # show the last captured photo's result while the camera is closed
    if image_bytes is None and st.session_state.get("captured_image"):
        image_bytes = st.session_state["captured_image"]
        image_name = "captured.jpg"

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


def render_analyzer() -> None:
    """The main tool: input controls and the analysis result."""
    image_bytes, image_name, eye_side, model = _collect_image()

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
    if not image_bytes:
        return

    # Detection is expensive — cache it per image+eye+model so that nudging only
    # redoes the cheap computation (overlay + IPR) in render_result, never the
    # full analysis again.
    cache_key = (hash(image_bytes), eye_side, model)
    cached = st.session_state.get("analysis_cache")
    if not cached or cached.get("key") != cache_key:
        with st.spinner("Analyzing…"):
            result = analysis.analyze_one(image_bytes, image_name, eye_side, model)
        st.session_state["analysis_cache"] = {"key": cache_key, "result": result}
    else:
        result = cached["result"]

    render_result(result, eye_side)


def render_about() -> None:
    """Static About page: what the app does, the output, authors and license."""
    st.markdown(
        """
        <div class="iris-card">
            <h3>What is Iris Analyzer?</h3>
            <p>Iris Analyzer is a tool for measuring the geometry of the human eye
            from near-infrared (IR) iris images. For a single uploaded or
            camera-captured image it removes specular reflections inside the pupil,
            runs the <b>open-iris</b> recognition pipeline to segment the eye,
            estimates the pupil and iris circles, and reports the
            <b>Iris-to-Pupil Ratio (IPR)</b> — a normalized measure of pupil
            dilation that is independent of image scale.</p>
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
    render_navbar(page)

    if page == "about":
        render_about()
    else:
        render_analyzer()

    render_footer()
