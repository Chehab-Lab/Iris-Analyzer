"""
Styling for the Iris Analyzer app.

Everything visual lives here: Streamlit page configuration and the single CSS
block that turns the default Streamlit chrome into a clean, light, desktop-style
white UI. No analysis or UI-flow logic.
"""

from __future__ import annotations

import streamlit as st


def configure_page() -> None:
    """Set the Streamlit page config. Must be the first Streamlit call."""
    st.set_page_config(
        page_title="Iris Analyzer",
        page_icon="\U0001F441",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_css() -> None:
    """Clean, light desktop-application UI on a white background."""
    st.markdown(
        """
        <style>
        :root {
            --bg:        #ffffff;
            --panel:     #ffffff;
            --panel-2:   #f7f8fa;
            --border:    #e5e7eb;
            --border-2:  #d1d5db;
            --text:      #1f2937;
            --text-dim:  #6b7280;
            --accent:    #2563eb;
            --accent-h:  #1d4ed8;
            --good:      #16a34a;
            --bad:       #dc2626;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         Helvetica, Arial, sans-serif;
        }

        /* Hide default Streamlit chrome */
        #MainMenu, header[data-testid="stHeader"], footer {visibility: hidden;}
        section[data-testid="stSidebar"] {display: none;}
        .block-container {padding-top: 5.2rem; max-width: 900px;}

        /* ---- Top nav bar ---- */
        .iris-nav {
            position: fixed; top: 0; left: 0; right: 0; z-index: 999;
            display: flex; align-items: center; gap: 11px;
            height: 56px; padding: 0 26px;
            background: #ffffff;
            border-bottom: 1px solid var(--border);
        }
        .iris-nav .glyph {font-size: 1.25rem;}
        .iris-nav .name {
            font-size: 1.12rem; font-weight: 600; letter-spacing: -.2px;
            color: var(--text);
        }

        /* ---- Cards ---- */
        .iris-card {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--panel);
            padding: 20px 22px;
            margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(16,24,40,.04);
        }
        .iris-card h3 {
            font-size: .95rem; font-weight: 600; letter-spacing: 0;
            color: var(--text); margin: 0 0 14px; text-transform: none;
        }

        /* ---- Metric tiles ---- */
        .metric-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px;}
        .metric {
            border:1px solid var(--border); border-radius:10px;
            background: var(--panel-2);
            padding:13px 15px;
        }
        .metric .label {color:var(--text-dim); font-size:.72rem; font-weight:500;}
        .metric .value {font-size:1.4rem; font-weight:600; color:var(--text); margin-top:3px;}
        .metric .unit  {color:var(--text-dim); font-size:.8rem; font-weight:400;}
        .metric.hl {background:#eff6ff; border-color:#bfdbfe;}
        .metric.hl .value {color:var(--accent);}

        /* ---- Buttons (flat, desktop-style) ---- */
        .stButton>button {
            font-weight:500; border-radius:8px; padding:.55rem 1rem;
            transition: background .12s ease, border-color .12s ease;
        }
        /* active source = solid blue */
        .stButton>button[kind="primary"] {
            background: var(--accent); color:#fff; border:1px solid var(--accent);
        }
        .stButton>button[kind="primary"]:hover,
        .stButton>button[kind="primary"]:focus {
            background: var(--accent-h); border-color: var(--accent-h); color:#fff; box-shadow:none;
        }
        /* inactive source = outline */
        .stButton>button[kind="secondary"] {
            background:#fff; color:var(--text); border:1px solid var(--border-2);
        }
        .stButton>button[kind="secondary"]:hover,
        .stButton>button[kind="secondary"]:focus {
            background:var(--panel-2); color:var(--text); border-color:var(--border-2); box-shadow:none;
        }

        .stDownloadButton>button {
            background:#ffffff; color:var(--text); font-weight:500;
            border:1px solid var(--border-2); border-radius:8px; padding:.55rem 1rem;
        }
        .stDownloadButton>button:hover {background:var(--panel-2); color:var(--text); border-color:var(--border-2);}

        /* segmented radio (the one setting) */
        div[role="radiogroup"] {gap:8px;}

        /* file uploader rendered as a single solid button (no dropzone box) */
        [data-testid="stFileUploader"] > label {display:none;}
        [data-testid="stFileUploaderDropzone"] {
            display:flex; align-items:center; justify-content:center;
            min-height:0; padding:.55rem 1rem; cursor:pointer;
            background:var(--accent); border:1px solid var(--accent); border-radius:8px;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            background:var(--accent-h); border-color:var(--accent-h);
        }
        /* hide ALL dropzone contents (icon, file chip, name, size, "+"),
           leaving only our label below */
        [data-testid="stFileUploaderDropzone"] * {display:none !important;}
        [data-testid="stFileUploaderDropzone"]::after {
            content:"Upload image"; color:#fff; font-weight:500; font-size:.9rem;
        }
        /* hide any file list rendered as a sibling of the dropzone */
        [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] ul {display:none !important;}

        /* dataframe */
        [data-testid="stDataFrame"] {border:1px solid var(--border); border-radius:10px;}

        /* images */
        [data-testid="stImage"] img {border:1px solid var(--border); border-radius:10px;}

        /* ---- Footer ---- */
        .iris-footer {
            display:flex; align-items:center; justify-content:center; gap:7px;
            margin:38px 0 14px; padding-top:18px;
            border-top:1px solid var(--border);
            color:var(--text-dim); font-size:.85rem;
        }
        .iris-footer .heart {color:#e0245e; font-size:.95rem;}
        .iris-footer a {display:inline-flex; align-items:center; text-decoration:none;}
        .iris-footer img {height:20px; width:auto; vertical-align:middle; border:none !important; cursor:pointer;}
        .iris-footer a:hover img {opacity:.8;}
        .iris-footer .lab {font-weight:600; color:var(--text);}
        </style>
        """,
        unsafe_allow_html=True,
    )
