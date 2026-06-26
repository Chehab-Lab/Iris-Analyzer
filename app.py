"""
Iris Analyzer — Streamlit entrypoint.

Thin wiring layer. The app is split into modules:
  * styles.py    — page config + CSS (all styling)
  * analysis.py  — engine bootstrap + iris/image processing (no UI)
  * ui.py        — nav bar, controls and result rendering (UI logic)

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys

# Ensure this app's own directory is first on the import path so the local
# modules (styles/ui/analysis) always resolve here, regardless of the cwd or any
# similarly named installed package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import styles
import ui

styles.configure_page()  # must be the first Streamlit call
styles.inject_css()
ui.run()
