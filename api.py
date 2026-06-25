"""
HTTP API for Iris Analyzer.

Exposes the same analysis engine the Streamlit UI uses (the shared ``analysis``
module) over a small JSON API, so other systems can analyze iris images
programmatically.

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /api/health   — liveness + whether the engine is ready
    POST /api/analyze  — analyze one image (multipart upload)
"""

from __future__ import annotations

import base64

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import analysis

app = FastAPI(
    title="Iris Analyzer API",
    version="1.0",
    description="Pupil/iris geometry and Iris-to-Pupil Ratio (IPR) from an eye image.",
)


@app.get("/api/health")
def health() -> dict:
    """Liveness check and engine readiness."""
    return {"status": "ok", "engine_ready": analysis.ENGINE_READY}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(..., description="Eye image (png/jpg/bmp/tif)."),
    eye_side: str = Form("right", description="'right' or 'left'."),
    include_overlay: bool = Form(False, description="Return the annotated overlay PNG."),
) -> JSONResponse:
    """Analyze a single eye image and return the measurements as JSON."""
    if not analysis.ENGINE_READY:
        raise HTTPException(status_code=503, detail="Analysis engine is not available.")
    if eye_side not in ("right", "left"):
        raise HTTPException(status_code=400, detail="eye_side must be 'right' or 'left'.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    result = analysis.analyze_one(data, image.filename or "upload", eye_side)
    if not result.ok:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "error": result.error or "Analysis failed."},
        )

    payload = {
        "status": "ok",
        "eye_side": eye_side,
        "ipr": result.ipr,
        "pupil_center": list(result.pupil_center),
        "pupil_radius": result.pupil_radius,
        "iris_center": list(result.iris_center),
        "iris_radius": result.iris_radius,
        "width": result.width,
        "height": result.height,
    }
    if include_overlay and result.overlay_png:
        payload["overlay_png_base64"] = base64.b64encode(result.overlay_png).decode("ascii")

    return JSONResponse(content=payload)
