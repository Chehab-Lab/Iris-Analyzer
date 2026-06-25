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

import analysis
from analysis import IrisResult

CHEHAB_LAB_URL = "https://chehablab.com"


@lru_cache(maxsize=1)
def _logo_data_uri() -> str:
    """Return the Chehab Lab logo as an inline base64 data URI (cached)."""
    try:
        logo = Path(__file__).parent / "assets" / "chehab-lab-logo.png"
        b64 = base64.b64encode(logo.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def render_navbar() -> None:
    """Fixed top bar with the app title."""
    st.markdown(
        """
        <div class="iris-nav">
            <span class="glyph">\U0001F441</span>
            <span class="name">Iris Analyzer</span>
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
                # opening the camera clears any previously uploaded image / result
                st.query_params["cam"] = "1"
                st.session_state["uploader_key"] += 1
            st.rerun()

    image_bytes: Optional[bytes] = None
    image_name = "captured.png"

    if up is not None:
        image_bytes = up.getvalue()
        image_name = up.name
        # uploading an image closes the live camera stream
        if cam_on:
            del st.query_params["cam"]
            st.rerun()

    if cam_on:
        shot = st.camera_input("Take a photo", label_visibility="collapsed", key="camera")
        if shot is not None:
            image_bytes = shot.getvalue()
            image_name = "captured.png"
        else:
            st.caption("After clicking **Allow** for camera access, reload the page — "
                       "the camera will stay open and start working.")

    return image_bytes, image_name, eye_side


def render_footer() -> None:
    """Bottom credit line linking to Chehab Lab."""
    logo = _logo_data_uri()
    badge = (f'<img src="{logo}" alt="Chehab Lab" />' if logo
             else '<span class="lab">Chehab Lab</span>')
    st.markdown(
        f'<a class="iris-footer" href="{CHEHAB_LAB_URL}" target="_blank" '
        f'rel="noopener noreferrer">Made with <span class="heart">❤</span> '
        f'by {badge} @ 2026</a>',
        unsafe_allow_html=True,
    )


def run() -> None:
    """Render the whole app: nav bar, controls, result, then the footer."""
    render_navbar()

    image_bytes, image_name, eye_side = _collect_image()

    if not analysis.ENGINE_READY:
        st.caption("Analysis engine is not available in this environment.")

    if image_bytes and analysis.ENGINE_READY:
        with st.spinner("Analyzing…"):
            result = analysis.analyze_one(image_bytes, image_name, eye_side)
        render_result(result, eye_side)

    render_footer()
