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


def render_result(r: IrisResult, eye_side: str) -> None:
    """Show the overlay, metric tiles and download buttons for one result."""
    if not r.ok:
        st.error(r.error or "Analysis failed.")
        if r.original_png:
            st.image(r.original_png, use_container_width=True, caption="Source image")
        return

    img_col, data_col = st.columns([3, 2], gap="large")
    with img_col:
        st.image(r.overlay_png, use_container_width=True,
                 caption="Pupil & iris boundaries")
    with data_col:
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric hl"><div class="label">IPR</div>
                    <div class="value">{r.ipr:.4f}</div></div>
                <div class="metric"><div class="label">Eye side</div>
                    <div class="value" style="font-size:1.1rem">{eye_side.title()}</div></div>
                <div class="metric"><div class="label">Iris radius</div>
                    <div class="value">{r.iris_radius:.1f}<span class="unit"> px</span></div></div>
                <div class="metric"><div class="label">Pupil radius</div>
                    <div class="value">{r.pupil_radius:.1f}<span class="unit"> px</span></div></div>
                <div class="metric"><div class="label">Iris center</div>
                    <div class="value" style="font-size:1rem">{r.iris_center[0]:.0f}, {r.iris_center[1]:.0f}</div></div>
                <div class="metric"><div class="label">Pupil center</div>
                    <div class="value" style="font-size:1rem">{r.pupil_center[0]:.0f}, {r.pupil_center[1]:.0f}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    df = analysis.result_to_df(r, eye_side)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download overlay (PNG)",
            data=r.overlay_png,
            file_name=f"iris_overlay_{stamp}.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download measurements (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"iris_result_{stamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _collect_image() -> tuple[Optional[bytes], str, str]:
    """Render the controls and return (image_bytes, image_name, eye_side).

    The camera's open/closed state lives in the URL (?cam=1) rather than in
    session_state: granting camera permission requires a page reload, which
    wipes session_state — but the URL survives, so the camera stays visible and
    works after the reload instead of disappearing.
    """
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("camera_key", 0)
    cam_on = st.query_params.get("cam") == "1"

    set_col, up_col, cam_col = st.columns([1.4, 1, 1])
    with set_col:
        eye_side = st.selectbox("Eye side", ["right", "left"], index=0)
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

    return image_bytes, image_name, eye_side


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
    image_bytes, image_name, eye_side = _collect_image()

    if not analysis.ENGINE_READY:
        st.caption("Analysis engine is not available in this environment.")

    if image_bytes and analysis.ENGINE_READY:
        with st.spinner("Analyzing…"):
            result = analysis.analyze_one(image_bytes, image_name, eye_side)
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
            <h3>API</h3>
            <p>The same engine is available as a JSON HTTP API served by the app
            itself, on the <b>same URL</b> as this page — so images can be analyzed
            programmatically without any separate backend.</p>
            <p><b>POST</b> <code>/api/analyze</code> — multipart form fields:
            <code>image</code> (file), <code>eye_side</code> (<code>right</code> or
            <code>left</code>), and optional <code>include_overlay</code>
            (<code>true</code>/<code>false</code>). <b>GET</b> <code>/api/health</code>
            reports whether the engine is ready.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        "curl -X POST https://<this-app-url>/api/analyze \\\n"
        "  -F image=@eye.png \\\n"
        "  -F eye_side=right",
        language="bash",
    )
    st.code(
        '{\n'
        '  "status": "ok",\n'
        '  "ipr": 2.83,\n'
        '  "pupil_radius": 41.2,\n'
        '  "iris_radius": 116.5,\n'
        '  "pupil_center": [312.0, 248.7],\n'
        '  "iris_center": [313.4, 249.1]\n'
        '}',
        language="json",
    )

    st.markdown(
        """
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
