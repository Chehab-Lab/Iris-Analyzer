"""
In-process HTTP API mounted onto Streamlit's own web server.

Streamlit serves the app with a Tornado web server. We attach our own route
handlers to that running server so the API lives on the *same* process and the
*same* URL as the UI (e.g. ``https://<app>.streamlit.app/api/analyze``). This is
what lets everything be hosted on Streamlit alone — no second server or port.

It relies on Streamlit/Tornado internals (private API), so it is written
defensively: if the internals change it silently no-ops and the UI is unaffected.

Endpoints:
    GET  /api/health   — liveness + engine readiness
    POST /api/analyze  — analyze one eye image (multipart field ``image``)
"""

from __future__ import annotations

import base64
import gc
import json

import tornado.web
from tornado.routing import PathMatches, Rule

import analysis

_MOUNTED = False


def _json(handler: tornado.web.RequestHandler, status: int, payload: dict) -> None:
    handler.set_status(status)
    handler.set_header("Content-Type", "application/json")
    handler.set_header("Access-Control-Allow-Origin", "*")
    handler.write(json.dumps(payload))


class _HealthHandler(tornado.web.RequestHandler):
    def check_xsrf_cookie(self) -> None:  # API is stateless; no XSRF
        pass

    def get(self) -> None:
        _json(self, 200, {"status": "ok", "engine_ready": analysis.ENGINE_READY})


class _AnalyzeHandler(tornado.web.RequestHandler):
    def check_xsrf_cookie(self) -> None:  # allow cross-origin POSTs without a token
        pass

    def set_default_headers(self) -> None:
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "*")

    def options(self) -> None:
        self.set_status(204)
        self.finish()

    def post(self) -> None:
        if not analysis.ENGINE_READY:
            return _json(self, 503, {"status": "error",
                                     "error": "Analysis engine is not available."})

        eye_side = self.get_argument("eye_side", "right")
        if eye_side not in ("right", "left"):
            return _json(self, 400, {"status": "error",
                                     "error": "eye_side must be 'right' or 'left'."})

        # image from a multipart 'image' field, or the raw request body
        files = self.request.files.get("image")
        if files:
            data, fname = files[0]["body"], files[0]["filename"] or "upload"
        else:
            data, fname = self.request.body, "upload"
        if not data:
            return _json(self, 400, {"status": "error", "error": "No image provided."})

        include_overlay = self.get_argument("include_overlay", "false").lower() in (
            "1", "true", "yes")

        try:
            result = analysis.analyze_one(data, fname, eye_side)
        except Exception as e:  # pragma: no cover
            return _json(self, 500, {"status": "error", "error": str(e)})

        if not result.ok:
            return _json(self, 422, {"status": "error",
                                     "error": result.error or "Analysis failed."})

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
            payload["overlay_png_base64"] = base64.b64encode(
                result.overlay_png).decode("ascii")
        _json(self, 200, payload)


def _find_streamlit_app():
    """Locate Streamlit's running Tornado Application (the one with the most rules)."""
    best, best_n = None, -1
    for obj in gc.get_objects():
        if isinstance(obj, tornado.web.Application):
            try:
                n = len(obj.wildcard_router.rules)
            except Exception:
                n = 0
            if n > best_n:
                best, best_n = obj, n
    return best


def mount_api() -> None:
    """Attach /api/* handlers to Streamlit's Tornado server (once per process)."""
    global _MOUNTED
    if _MOUNTED:
        return
    try:
        app = _find_streamlit_app()
        if app is None:
            return
        new_rules = [
            Rule(PathMatches(r"/api/health"), _HealthHandler),
            Rule(PathMatches(r"/api/analyze"), _AnalyzeHandler),
        ]
        # Prepend so our routes win over Streamlit's catch-all static handler.
        app.wildcard_router.rules[0:0] = new_rules
        _MOUNTED = True
    except Exception:
        pass
