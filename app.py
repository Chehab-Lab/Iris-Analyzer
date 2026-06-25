"""
Iris Analyzer — Streamlit application
=====================================
Batch analysis of IR iris images. For every uploaded image the app:
  * cleans specular reflections inside the pupil,
  * runs the `iris` recognition pipeline,
  * estimates pupil / iris geometry and the Iris-to-Pupil Ratio (IPR),
  * renders an annotated overlay,
and lets the user download all overlays + a results.csv as a single ZIP.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless backend — required inside Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Optional heavy dependencies. We import lazily / defensively so the UI can
# still load (and explain itself) on a machine where `iris` or `cv2` are not
# yet installed.
# ----------------------------------------------------------------------------
try:
    import cv2

    _CV2_OK = True
except Exception:  # pragma: no cover
    _CV2_OK = False

try:
    import iris

    _IRIS_OK = True
except Exception:  # pragma: no cover
    _IRIS_OK = False


# ============================================================================
#  Page configuration & global styling
# ============================================================================
st.set_page_config(
    page_title="Iris Analyzer",
    page_icon="\U0001F441",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    """Custom, scientific dark UI — overrides the default Streamlit chrome."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg:        #0b0f17;
            --bg-soft:   #111725;
            --panel:     #151c2c;
            --panel-2:   #1b2438;
            --border:    #243049;
            --text:      #e6ecf5;
            --text-dim:  #8b97ac;
            --accent:    #4dd0e1;
            --accent-2:  #7c83ff;
            --good:      #34d399;
            --warn:      #fbbf24;
            --bad:       #f87171;
        }

        .stApp {
            background:
                radial-gradient(1200px 600px at 12% -8%, rgba(124,131,255,.10), transparent 60%),
                radial-gradient(1000px 600px at 110% 0%, rgba(77,208,225,.08), transparent 55%),
                var(--bg);
            color: var(--text);
            font-family: 'Inter', system-ui, sans-serif;
        }

        /* Hide default Streamlit chrome */
        #MainMenu, header[data-testid="stHeader"], footer {visibility: hidden;}
        .block-container {padding-top: 1.4rem; max-width: 1320px;}

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--bg-soft), var(--bg));
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] * {color: var(--text);}

        /* ---- Hero header ---- */
        .iris-hero {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 26px 30px;
            background:
                linear-gradient(135deg, rgba(124,131,255,.10), rgba(77,208,225,.04)),
                var(--panel);
            box-shadow: 0 18px 50px -30px rgba(0,0,0,.9);
            margin-bottom: 22px;
        }
        .iris-hero h1 {
            font-size: 1.85rem; font-weight: 700; margin: 0;
            letter-spacing: -.5px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .iris-hero p {color: var(--text-dim); margin: 8px 0 0; font-size: .95rem;}
        .iris-chip {
            display:inline-flex; align-items:center; gap:7px;
            font-family:'JetBrains Mono', monospace; font-size:.72rem;
            color: var(--accent); border:1px solid var(--border);
            background: rgba(77,208,225,.06);
            padding:4px 11px; border-radius:999px; margin-right:8px;
        }
        .dot {width:7px;height:7px;border-radius:50%;display:inline-block;}
        .dot.on{background:var(--good);box-shadow:0 0 8px var(--good);}
        .dot.off{background:var(--bad);box-shadow:0 0 8px var(--bad);}

        /* ---- Cards ---- */
        .iris-card {
            border: 1px solid var(--border);
            border-radius: 16px;
            background: var(--panel);
            padding: 18px 20px;
            margin-bottom: 16px;
        }
        .iris-card h3 {
            font-size: .82rem; text-transform: uppercase; letter-spacing: 1.4px;
            color: var(--text-dim); font-weight: 600; margin: 0 0 4px;
        }

        /* ---- Metric tiles ---- */
        .metric-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(135px,1fr)); gap:12px;}
        .metric {
            border:1px solid var(--border); border-radius:13px;
            background: linear-gradient(180deg, var(--panel-2), var(--panel));
            padding:14px 16px;
        }
        .metric .label {color:var(--text-dim); font-size:.7rem; text-transform:uppercase; letter-spacing:1px;}
        .metric .value {font-family:'JetBrains Mono',monospace; font-size:1.45rem; font-weight:600; color:var(--text); margin-top:4px;}
        .metric .unit  {color:var(--text-dim); font-size:.8rem; font-weight:400;}
        .metric.hl .value {color:var(--accent);}

        /* status pills */
        .pill {display:inline-block; padding:3px 10px; border-radius:999px; font-size:.72rem; font-weight:600; font-family:'JetBrains Mono',monospace;}
        .pill.ok  {background:rgba(52,211,153,.12); color:var(--good); border:1px solid rgba(52,211,153,.3);}
        .pill.err {background:rgba(248,113,113,.12); color:var(--bad); border:1px solid rgba(248,113,113,.3);}

        /* ---- Buttons ---- */
        .stButton>button, .stDownloadButton>button {
            background: linear-gradient(90deg, var(--accent-2), var(--accent));
            color:#0b0f17; font-weight:600; border:none; border-radius:11px;
            padding:.6rem 1.1rem; transition: transform .12s ease, box-shadow .12s ease;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px -12px rgba(77,208,225,.7);
            color:#0b0f17;
        }

        /* uploader */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--panel); border:1.5px dashed var(--border); border-radius:14px;
        }

        /* tabs */
        .stTabs [data-baseweb="tab-list"] {gap:6px; border-bottom:1px solid var(--border);}
        .stTabs [data-baseweb="tab"] {color:var(--text-dim); font-weight:500;}
        .stTabs [aria-selected="true"] {color:var(--accent);}

        /* dataframe */
        [data-testid="stDataFrame"] {border:1px solid var(--border); border-radius:12px;}

        /* progress bar */
        .stProgress > div > div > div > div {background: linear-gradient(90deg,var(--accent-2),var(--accent));}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ============================================================================
#  Data model
# ============================================================================
@dataclass
class IrisResult:
    name: str                       # cleaned base name of the source image
    ok: bool = False
    error: Optional[str] = None
    overlay_png: Optional[bytes] = None
    original_png: Optional[bytes] = None
    ipr: float = float("nan")
    pupil_center: tuple = field(default_factory=lambda: (float("nan"), float("nan")))
    pupil_radius: float = float("nan")
    iris_center: tuple = field(default_factory=lambda: (float("nan"), float("nan")))
    iris_radius: float = float("nan")
    width: int = 0
    height: int = 0


# ============================================================================
#  Core analysis  (faithful port of the original FastAPI logic)
# ============================================================================
def _render_overlay(img_cleaned, pupil_xy, pupil_radius, iris_contour, iris_center) -> bytes:
    """Build the annotated matplotlib overlay and return it as PNG bytes."""
    fig = plt.figure(figsize=(8, 8), facecolor="#0b0f17")
    ax = plt.gca()
    ax.imshow(img_cleaned, cmap="gray")

    pupil_x, pupil_y = pupil_xy
    ax.add_patch(
        plt.Circle((pupil_x, pupil_y), pupil_radius, fill=False,
                   color="#34d399", linewidth=2, label="Pupil circle")
    )
    ax.plot(iris_contour[:, 0], iris_contour[:, 1], color="#4dd0e1",
            linewidth=2, label="Iris boundary")
    ax.plot(iris_center[0], iris_center[1], "o", color="#7c83ff",
            markersize=6, label="Iris center")
    ax.plot(pupil_x, pupil_y, "o", color="#f87171",
            markersize=6, label="Pupil center")

    ax.set_title("Pupil & Iris boundaries", color="#e6ecf5")
    leg = ax.legend(facecolor="#151c2c", edgecolor="#243049", labelcolor="#e6ecf5")
    for txt in leg.get_texts():
        txt.set_color("#e6ecf5")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200,
                facecolor="#0b0f17")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def analyze_iris_image(img_pixels, eye_side: str = "right") -> dict:
    """Reflection removal + IRIS pipeline + geometry. Mirrors the API version."""
    try:
        # Step 1 — initial pass for segmentation
        try:
            iris_pipeline_temp = iris.IRISPipeline(env=iris.IRISPipeline.DEBUGGING_ENVIRONMENT)
            _ = iris_pipeline_temp(img_data=img_pixels, eye_side=eye_side)
        except Exception as e:
            return {"error": f"Failed to initialize IRIS pipeline: {e}"}

        # Step 2 — pupil segmentation mask
        try:
            segmap = iris_pipeline_temp.call_trace["segmentation"].predictions
            pupil_softmax = segmap[:, :, 2]
            pupil_mask = (pupil_softmax > 0.5).astype(np.uint8) * 255
        except Exception as e:
            return {"error": f"Failed to extract pupil segmentation: {e}"}

        # Step 3 — bright reflections inside pupil
        try:
            reflection_threshold = 200
            bright_spots = (img_pixels > reflection_threshold).astype(np.uint8) * 255
            reflection_mask = cv2.bitwise_and(bright_spots, pupil_mask)
        except Exception as e:
            return {"error": f"Failed to detect reflections: {e}"}

        # Step 3b — dilate mask
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            reflection_mask = cv2.dilate(reflection_mask, kernel, iterations=3)
        except Exception as e:
            return {"error": f"Failed to refine reflection mask: {e}"}

        # Step 4 — inpaint
        try:
            img_cleaned = cv2.inpaint(img_pixels, reflection_mask,
                                      inpaintRadius=9, flags=cv2.INPAINT_NS)
        except Exception as e:
            return {"error": f"Failed to inpaint reflections: {e}"}

        # Step 5 — final pipeline on cleaned image
        try:
            iris_pipeline = iris.IRISPipeline(env=iris.IRISPipeline.DEBUGGING_ENVIRONMENT)
            output = iris_pipeline(img_data=img_cleaned, eye_side=eye_side)
        except Exception as e:
            return {"error": f"Failed to run final IRIS pipeline: {e}"}

        try:
            geometry = iris_pipeline.call_trace["geometry_estimation"]
            if geometry is None:
                return {"error": "Geometry estimation failed."}
        except Exception as e:
            return {"error": f"Failed to access geometry estimation: {e}"}

        # Pupil measurements
        try:
            pupil_array = geometry.pupil_array
            pupil_contour = pupil_array.reshape(-1, 1, 2).astype(np.float32)
            (pupil_x, pupil_y), pupil_radius = cv2.minEnclosingCircle(pupil_contour)
        except Exception as e:
            return {"error": f"Failed to calculate pupil measurements: {e}"}

        # Iris measurements
        try:
            iris_center = output["metadata"]["eye_centers"]["iris_center"]
            iris_radius = (output["metadata"]["iris_bbox"]["x_max"]
                           - output["metadata"]["iris_bbox"]["x_min"]) / 2
        except Exception as e:
            return {"error": f"Failed to calculate iris measurements: {e}"}

        ipr = iris_radius / pupil_radius if pupil_radius > 0 else 0.0

        # Overlay
        try:
            iris_contour = geometry.iris_array.reshape(-1, 2)
            overlay_png = _render_overlay(img_cleaned, (pupil_x, pupil_y),
                                          pupil_radius, iris_contour, iris_center)
        except Exception as e:
            return {"error": f"Failed to create visualization: {e}"}

        return {
            "ipr": float(ipr),
            "overlay_png": overlay_png,
            "pupil_center": [float(pupil_x), float(pupil_y)],
            "pupil_radius": float(pupil_radius),
            "iris_center": [float(iris_center[0]), float(iris_center[1])],
            "iris_radius": float(iris_radius),
        }
    except Exception as e:
        return {"error": f"Unexpected error in iris analysis: {e}"}


# ============================================================================
#  Image loading helper
# ============================================================================
def load_grayscale(file_bytes: bytes, max_dimension: int = 800):
    """Decode → grayscale → smart-resize. Returns (img, w, h) or raises."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid or unsupported image file.")
    h, w = img.shape[:2]
    if h > max_dimension or w > max_dimension:
        scale = max_dimension / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    return img, w, h


def png_bytes_from_gray(img) -> bytes:
    ok, enc = cv2.imencode(".png", img)
    return enc.tobytes() if ok else b""


# ============================================================================
#  ZIP packaging
# ============================================================================
def safe_stem(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
    return keep.strip("_") or "image"


def build_results_df(results: list[IrisResult]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results, 1):
        rows.append({
            "index": i,
            "image": r.name,
            "status": "ok" if r.ok else "error",
            "error": r.error or "",
            "ipr": round(r.ipr, 5) if r.ok else "",
            "pupil_x": round(r.pupil_center[0], 3) if r.ok else "",
            "pupil_y": round(r.pupil_center[1], 3) if r.ok else "",
            "pupil_radius": round(r.pupil_radius, 3) if r.ok else "",
            "iris_x": round(r.iris_center[0], 3) if r.ok else "",
            "iris_y": round(r.iris_center[1], 3) if r.ok else "",
            "iris_radius": round(r.iris_radius, 3) if r.ok else "",
            "width": r.width,
            "height": r.height,
        })
    return pd.DataFrame(rows)


def build_zip(results: list[IrisResult], df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, r in enumerate(results, 1):
            base = f"{i:02d}_{safe_stem(r.name)}"
            if r.ok and r.overlay_png:
                zf.writestr(f"Images/{base}.png", r.overlay_png)
            elif r.original_png:
                # still ship the source so the folder is complete
                zf.writestr(f"Images/{base}_FAILED.png", r.original_png)
        zf.writestr("results.csv", df.to_csv(index=False))
        zf.writestr(
            "README.txt",
            "Iris Analyzer export\n"
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"Images analyzed: {len(results)}\n\n"
            "Images/  - annotated overlays (one per input image)\n"
            "results.csv - per-image measurements (IPR, geometry, status)\n",
        )
    buf.seek(0)
    return buf.read()


# ============================================================================
#  Sidebar
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️  Configuration")
    eye_side = st.radio("Eye side", ["right", "left"], horizontal=True,
                        help="Side passed to the IRIS pipeline.")
    max_dim = st.slider("Max image dimension (px)", 400, 1600, 800, 50,
                        help="Larger images are downscaled before analysis.")
    st.divider()
    st.markdown("### \U0001F9EA  Engine status")
    st.markdown(
        f"<span class='pill {'ok' if _IRIS_OK else 'err'}'>"
        f"iris {'ready' if _IRIS_OK else 'missing'}</span> &nbsp;"
        f"<span class='pill {'ok' if _CV2_OK else 'err'}'>"
        f"opencv {'ready' if _CV2_OK else 'missing'}</span>",
        unsafe_allow_html=True,
    )
    if not (_IRIS_OK and _CV2_OK):
        st.caption("Install dependencies: `pip install -r requirements.txt`")
    st.divider()
    st.caption("Iris Analyzer · scientific batch tool")


# ============================================================================
#  Hero
# ============================================================================
st.markdown(
    f"""
    <div class="iris-hero">
        <h1>Iris Analyzer</h1>
        <p>Batch reflection-removal, segmentation and Iris-to-Pupil-Ratio (IPR)
           estimation for near-infrared iris imagery.</p>
        <div style="margin-top:14px;">
            <span class="iris-chip">
                <span class="dot {'on' if _IRIS_OK else 'off'}"></span> IRIS pipeline
            </span>
            <span class="iris-chip">
                <span class="dot {'on' if _CV2_OK else 'off'}"></span> OpenCV inpainting
            </span>
            <span class="iris-chip">IPR = iris_radius / pupil_radius</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
#  Upload
# ============================================================================
st.markdown('<div class="iris-card"><h3>1 · Upload iris images</h3>',
            unsafe_allow_html=True)
files = st.file_uploader(
    "Drop one or more IR iris images",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
col_a, col_b = st.columns([1, 3])
with col_a:
    run = st.button("▶  Analyze", use_container_width=True,
                    disabled=not files or not (_IRIS_OK and _CV2_OK))
with col_b:
    if files:
        st.markdown(
            f"<div style='padding-top:.55rem;color:var(--text-dim);'>"
            f"{len(files)} image(s) queued · eye side <b>{eye_side}</b></div>",
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
#  Run analysis
# ============================================================================
if run and files:
    results: list[IrisResult] = []
    progress = st.progress(0.0, text="Starting…")
    for idx, f in enumerate(files, 1):
        progress.progress(idx / len(files), text=f"Analyzing {f.name} ({idx}/{len(files)})")
        res = IrisResult(name=f.name)
        try:
            img, w, h = load_grayscale(f.getvalue(), max_dimension=max_dim)
            res.width, res.height = w, h
            res.original_png = png_bytes_from_gray(img)
            out = analyze_iris_image(img, eye_side=eye_side)
            if "error" in out:
                res.error = out["error"]
            else:
                res.ok = True
                res.overlay_png = out["overlay_png"]
                res.ipr = out["ipr"]
                res.pupil_center = tuple(out["pupil_center"])
                res.pupil_radius = out["pupil_radius"]
                res.iris_center = tuple(out["iris_center"])
                res.iris_radius = out["iris_radius"]
        except Exception as e:
            res.error = str(e)
        results.append(res)
    progress.empty()
    st.session_state["results"] = results


# ============================================================================
#  Results
# ============================================================================
results: list[IrisResult] = st.session_state.get("results", [])

if results:
    ok_n = sum(r.ok for r in results)
    err_n = len(results) - ok_n
    iprs = [r.ipr for r in results if r.ok and np.isfinite(r.ipr)]
    mean_ipr = float(np.mean(iprs)) if iprs else float("nan")

    # ---- summary band ----
    st.markdown('<div class="iris-card"><h3>2 · Summary</h3>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric"><div class="label">Images</div>
                <div class="value">{len(results)}</div></div>
            <div class="metric"><div class="label">Succeeded</div>
                <div class="value" style="color:var(--good)">{ok_n}</div></div>
            <div class="metric"><div class="label">Failed</div>
                <div class="value" style="color:{'var(--bad)' if err_n else 'var(--text)'}">{err_n}</div></div>
            <div class="metric hl"><div class="label">Mean IPR</div>
                <div class="value">{mean_ipr:.3f}{'' if iprs else ' <span class=unit>n/a</span>'}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- per-image tabs ----
    st.markdown('<div class="iris-card"><h3>3 · Per-image results</h3>',
                unsafe_allow_html=True)
    tabs = st.tabs([f"{i:02d} · {r.name}" for i, r in enumerate(results, 1)])
    for tab, r in zip(tabs, results):
        with tab:
            if r.ok:
                c1, c2 = st.columns([3, 2], gap="large")
                with c1:
                    st.image(r.overlay_png, use_container_width=True,
                             caption="Annotated overlay")
                with c2:
                    st.markdown(
                        f"""
                        <div class="metric-grid">
                            <div class="metric hl"><div class="label">IPR</div>
                                <div class="value">{r.ipr:.4f}</div></div>
                            <div class="metric"><div class="label">Iris radius</div>
                                <div class="value">{r.iris_radius:.1f}<span class="unit"> px</span></div></div>
                            <div class="metric"><div class="label">Pupil radius</div>
                                <div class="value">{r.pupil_radius:.1f}<span class="unit"> px</span></div></div>
                            <div class="metric"><div class="label">Iris center</div>
                                <div class="value" style="font-size:1rem">{r.iris_center[0]:.0f}, {r.iris_center[1]:.0f}</div></div>
                            <div class="metric"><div class="label">Pupil center</div>
                                <div class="value" style="font-size:1rem">{r.pupil_center[0]:.0f}, {r.pupil_center[1]:.0f}</div></div>
                            <div class="metric"><div class="label">Resolution</div>
                                <div class="value" style="font-size:1rem">{r.width}×{r.height}</div></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f"<span class='pill err'>analysis failed</span>",
                    unsafe_allow_html=True,
                )
                st.error(r.error or "Unknown error")
                if r.original_png:
                    st.image(r.original_png, width=320, caption="Source image")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- results table ----
    df = build_results_df(results)
    st.markdown('<div class="iris-card"><h3>4 · Results table</h3>',
                unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- download ----
    st.markdown('<div class="iris-card"><h3>5 · Export</h3>', unsafe_allow_html=True)
    zip_bytes = build_zip(results, df)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cda, cdb = st.columns(2)
    with cda:
        st.download_button(
            "⬇  Download ZIP (images + results.csv)",
            data=zip_bytes,
            file_name=f"iris_analysis_{stamp}.zip",
            mime="application/zip",
            use_container_width=True,
        )
    with cdb:
        st.download_button(
            "⬇  Download results.csv only",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"iris_results_{stamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.caption("ZIP layout:  Images/NN_name.png  +  results.csv  +  README.txt")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        "<div class='iris-card' style='text-align:center;color:var(--text-dim);'>"
        "Upload images and press <b>Analyze</b> to see overlays, metrics and the "
        "downloadable export here.</div>",
        unsafe_allow_html=True,
    )
