"""
Iris Analyzer — Streamlit entrypoint.

Thin wiring layer. The app is split into modules:
  * styles.py      — page config + CSS (all styling)
  * analysis.py    — engine bootstrap + iris/image processing (no UI)
  * ui.py          — nav bar, controls and result rendering (UI logic)
  * api_routes.py  — /api/* HTTP endpoints mounted onto Streamlit's own server

Run with:  streamlit run app.py
"""

from __future__ import annotations

import api_routes
import styles
import ui

api_routes.mount_api()   # attach /api/* to the running Streamlit web server
styles.configure_page()  # must be the first Streamlit call
styles.inject_css()
ui.run()
