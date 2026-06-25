"""
Analysis logic for the Iris Analyzer app.

This module is pure logic — it imports no Streamlit and renders no UI. It covers:
  * runtime bootstrapping of the heavy native deps (onnxruntime exec-stack fix,
    open-iris model-cache redirect) so `iris` / `cv2` import on hardened hosts,
  * the IRIS recognition pipeline (reflection removal → segmentation → geometry),
  * the `IrisResult` data model and a single-image `analyze_one` entry point,
  * CSV serialization of a result.

Public surface used by the UI: CV2_OK, IRIS_OK, IRIS_ERR, ENGINE_READY, MAX_DIM,
IrisResult, analyze_one, result_to_df.
"""

from __future__ import annotations

import io
import os
import shutil
import struct
import sys
import sysconfig
import tempfile
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless backend — required inside Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAX_DIM = 1000  # fixed internal downscale guard (no user-facing setting)

# ----------------------------------------------------------------------------
# Optional heavy dependencies. We import lazily / defensively so the app can
# still load on a machine where `iris` or `cv2` are not yet installed.
# ----------------------------------------------------------------------------
try:
    import cv2

    CV2_OK = True
except Exception:  # pragma: no cover
    cv2 = None
    CV2_OK = False

_EXECSTACK_LOG: list[str] = []


def _patch_so_execstack(so: str) -> str:
    """Clear PT_GNU_STACK's executable bit on one ELF64 .so. Returns a status."""
    PT_GNU_STACK = 0x6474E551
    PF_X = 0x1
    try:
        with open(so, "rb") as f:
            data = bytearray(f.read())
    except Exception as e:
        return f"read-fail ({e.__class__.__name__})"
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return "not-elf64"
    e_phoff = struct.unpack_from("<Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x36)[0]
    e_phnum = struct.unpack_from("<H", data, 0x38)[0]
    changed = False
    found = False
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        if struct.unpack_from("<I", data, off)[0] != PT_GNU_STACK:
            continue
        found = True
        p_flags = struct.unpack_from("<I", data, off + 4)[0]
        if p_flags & PF_X:
            struct.pack_into("<I", data, off + 4, p_flags & ~PF_X)
            changed = True
    if not found:
        return "no-gnu-stack"
    if not changed:
        return "already-clean"
    # Persist the patched bytes (make writable first if needed).
    try:
        try:
            os.chmod(so, 0o755)
        except Exception:
            pass
        with open(so, "rb+") as f:
            f.seek(0)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return "patched-ok"
    except Exception as e:
        # Read-only install dir: rewrite atomically via a temp file + replace.
        try:
            tmp = so + ".patched"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, so)
            return "patched-via-replace"
        except Exception as e2:
            return f"write-fail ({e.__class__.__name__}/{e2.__class__.__name__})"


def _onnxruntime_pkg_dir() -> Optional[str]:
    roots = {
        sysconfig.get_paths().get("purelib", ""),
        sysconfig.get_paths().get("platlib", ""),
        os.path.normpath(os.path.join(os.path.dirname(np.__file__), os.pardir)),
    }
    for root in filter(None, roots):
        cand = os.path.join(root, "onnxruntime")
        if os.path.isdir(cand):
            return cand
    return None


def _clear_execstack_flag() -> None:
    """Ensure the onnxruntime native extension has no executable stack.

    onnxruntime==1.16.3 (hard-pinned by open-iris) ships its pybind .so with
    PT_GNU_STACK marked executable, which hardened kernels (Streamlit Cloud)
    refuse to dlopen. On Streamlit Cloud the install dir is read-only at
    runtime, so we cannot patch in place. Strategy:

      1. Try patching in place (works on writable installs).
      2. If that fails (read-only), copy the onnxruntime package into a
         writable temp dir, patch the copy, and prepend it to sys.path so the
         patched copy is imported instead of the read-only original.
    """
    src_pkg = _onnxruntime_pkg_dir()
    if not src_pkg:
        _EXECSTACK_LOG.append("onnxruntime package dir not found")
        return

    # 1) in-place attempt
    inplace = {}
    for so in glob(os.path.join(src_pkg, "capi", "*.so")):
        inplace[os.path.basename(so)] = _patch_so_execstack(so)
    for k, v in inplace.items():
        _EXECSTACK_LOG.append(f"inplace {k}: {v}")

    if not any("write-fail" in v for v in inplace.values()):
        return  # already clean or successfully patched in place

    # 2) writable shadow copy + sys.path shim
    try:
        shadow_root = os.path.join(tempfile.gettempdir(), "ort_shadow")
        dst_pkg = os.path.join(shadow_root, "onnxruntime")
        marker = os.path.join(dst_pkg, ".execstack_patched")
        if not os.path.exists(marker):
            if os.path.isdir(dst_pkg):
                shutil.rmtree(dst_pkg, ignore_errors=True)
            os.makedirs(shadow_root, exist_ok=True)
            shutil.copytree(src_pkg, dst_pkg)
            for so in glob(os.path.join(dst_pkg, "capi", "*.so")):
                _EXECSTACK_LOG.append(f"shadow {os.path.basename(so)}: {_patch_so_execstack(so)}")
            with open(marker, "w") as m:
                m.write("ok")
        if shadow_root not in sys.path:
            sys.path.insert(0, shadow_root)
        _EXECSTACK_LOG.append(f"shadow active: {shadow_root}")
    except Exception as e:
        _EXECSTACK_LOG.append(f"shadow-fail: {e.__class__.__name__}: {e}")


def _redirect_iris_model_cache() -> None:
    """Point open-iris's segmentation-model cache at a writable directory.

    open-iris caches its HuggingFace segmentation ONNX in
    ``MODEL_CACHE_DIR = <iris pkg>/nodes/segmentation/assets``. On Streamlit
    Cloud that site-packages tree is read-only, so ``hf_hub_download`` raises
    ``PermissionError`` ("...segmentation/assets"). We copy whatever is already
    bundled there into a writable temp dir (so no re-download is needed when the
    model ships with the wheel) and repoint the class attribute at it.
    """
    try:
        from iris.nodes.segmentation import (
            multilabel_segmentation_interface as _msi,
        )

        cls = _msi.MultilabelSemanticSegmentationInterface
        src = getattr(cls, "MODEL_CACHE_DIR", None)
        dst = os.path.join(tempfile.gettempdir(), "iris_model_cache")
        if not os.path.isdir(dst):
            try:
                if src and os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    os.makedirs(dst, exist_ok=True)
            except Exception:
                os.makedirs(dst, exist_ok=True)
        cls.MODEL_CACHE_DIR = dst
    except Exception:
        pass


IRIS_ERR = ""
try:
    _clear_execstack_flag()
    import iris

    _redirect_iris_model_cache()
    IRIS_OK = True
except Exception as _e:  # pragma: no cover
    iris = None
    IRIS_OK = False
    import traceback as _tb

    IRIS_ERR = "".join(_tb.format_exception_only(type(_e), _e)).strip()

ENGINE_READY = IRIS_OK and CV2_OK


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
    fig = plt.figure(figsize=(8, 8), facecolor="#ffffff")
    ax = plt.gca()
    ax.imshow(img_cleaned, cmap="gray")

    pupil_x, pupil_y = pupil_xy
    ax.add_patch(
        plt.Circle((pupil_x, pupil_y), pupil_radius, fill=False,
                   color="#16a34a", linewidth=2, label="Pupil circle")
    )
    ax.plot(iris_contour[:, 0], iris_contour[:, 1], color="#2563eb",
            linewidth=2, label="Iris boundary")
    ax.plot(iris_center[0], iris_center[1], "o", color="#7c3aed",
            markersize=6, label="Iris center")
    ax.plot(pupil_x, pupil_y, "o", color="#dc2626",
            markersize=6, label="Pupil center")

    ax.set_title("Pupil & Iris boundaries", color="#1f2937")
    leg = ax.legend(facecolor="#ffffff", edgecolor="#e5e7eb", labelcolor="#1f2937")
    for txt in leg.get_texts():
        txt.set_color("#1f2937")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200,
                facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _pipeline_error_reason(output, pipeline) -> str:
    """Extract the real reason open-iris rejected an image.

    When a validation/geometry node fails, open-iris does not raise — it records
    the cause in the pipeline output's ``error`` field (and the call trace) and
    leaves downstream nodes as ``None``. We surface that instead of a generic
    "Geometry estimation failed" message.
    """
    # 1) the serialized output's error dict
    try:
        err = output.get("error") if isinstance(output, dict) else None
    except Exception:
        err = None
    if isinstance(err, dict):
        etype = err.get("error_type") or err.get("type") or ""
        msg = err.get("message") or ""
        text = f"{etype}: {msg}".strip(": ").strip()
        if text:
            return text
    elif err:
        return str(err)

    # 2) the call-trace error object
    try:
        getter = getattr(pipeline.call_trace, "get_error", None)
        tb_err = getter() if getter else None
        if tb_err is not None:
            return f"{type(tb_err).__name__}: {tb_err}"
    except Exception:
        pass

    return ("The image may not be a clear near-infrared iris, the eye side may be "
            "wrong, or the eye is too off-center / low-resolution.")


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
                reason = _pipeline_error_reason(output, iris_pipeline)
                return {"error": f"Could not locate the iris/pupil in this image. {reason}"}
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
#  Image loading helpers
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
#  Single-image entry point
# ============================================================================
def analyze_one(image_bytes: bytes, name: str, eye_side: str) -> IrisResult:
    """Decode + analyze one image and return a populated IrisResult."""
    res = IrisResult(name=name)
    try:
        img, w, h = load_grayscale(image_bytes, max_dimension=MAX_DIM)
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
    return res


# ============================================================================
#  CSV export
# ============================================================================
def result_to_df(r: IrisResult, eye_side: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "image": r.name,
        "eye_side": eye_side,
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
    }])
