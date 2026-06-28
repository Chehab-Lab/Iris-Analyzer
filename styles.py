"""
Styling for the Iris Analyzer app.

Everything visual lives here: Streamlit page configuration and the CSS
block that styles the app with a refined, premium light theme.
"""

from __future__ import annotations

import streamlit as st


def configure_page() -> None:
    """Set the Streamlit page config. Must be the first Streamlit call."""
    st.set_page_config(
        page_title="Pupillometry",
        page_icon="\U0001F441",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_css() -> None:
    """Refined, premium light UI — polished surfaces, depth, and typography."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg:           #eef1f7;
            --bg-soft:      #f7f8fc;
            --surface:      #ffffff;
            --surface-soft: #f4f6fb;
            --border:       rgba(15, 23, 42, 0.08);
            --border-2:     rgba(15, 23, 42, 0.14);
            --text:         #0f172a;
            --text-dim:     #64748b;
            --text-muted:   #94a3b8;
            --accent:       #2563eb;
            --accent-h:     #1d4ed8;
            --accent-soft:  #eff6ff;
            --accent-glow:  rgba(37, 99, 235, 0.18);
            --good:         #059669;
            --bad:          #dc2626;
            --shadow-xs:    0 1px 2px rgba(15, 23, 42, 0.04);
            --shadow-sm:    0 2px 8px rgba(15, 23, 42, 0.06);
            --shadow-md:    0 8px 24px rgba(15, 23, 42, 0.08);
            --shadow-lg:    0 18px 40px rgba(15, 23, 42, 0.10);
            --radius-sm:    10px;
            --radius-md:    14px;
            --radius-lg:    18px;
        }

        .stApp {
            background:
                radial-gradient(ellipse 120% 80% at 50% -20%, rgba(37, 99, 235, 0.07), transparent 55%),
                linear-gradient(180deg, var(--bg-soft) 0%, var(--bg) 100%);
            color: var(--text);
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         Helvetica, Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        /* Hide default Streamlit chrome */
        #MainMenu, header[data-testid="stHeader"], footer {visibility: hidden;}
        section[data-testid="stSidebar"] {display: none;}
        .block-container {
            padding-top: 5.6rem;
            max-width: 920px;
            padding-bottom: 2rem;
        }
        .block-container:has(.work-screen) {padding-bottom: 0.75rem;}
        .block-container:has(.work-screen) .iris-footer {margin-top: 20px;}

        /* ---- Top nav bar ---- */
        .iris-nav {
            position: fixed; top: 0; left: 0; right: 0; z-index: 999;
            display: flex; align-items: center; gap: 14px;
            height: 60px; padding: 0 28px;
            background: rgba(255, 255, 255, 0.82);
            backdrop-filter: blur(16px) saturate(1.4);
            -webkit-backdrop-filter: blur(16px) saturate(1.4);
            border-bottom: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }
        .iris-nav .glyph {
            display: flex; align-items: center; justify-content: center;
            width: 36px; height: 36px;
            font-size: 1.1rem;
            background: linear-gradient(145deg, var(--accent-soft), #fff);
            border: 1px solid rgba(37, 99, 235, 0.15);
            border-radius: 11px;
            box-shadow: var(--shadow-xs);
        }
        .iris-nav .nav-brand {
            display: flex; flex-direction: column; gap: 1px;
        }
        .iris-nav .name {
            font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em;
            color: var(--text); line-height: 1.2;
        }
        .iris-nav .tagline {
            font-size: 0.68rem; font-weight: 500; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--text-muted);
        }
        .iris-nav .nav-links {margin-left: auto; display: flex; gap: 8px;}
        .iris-nav .nav-links a {
            color: var(--text-dim); text-decoration: none; font-size: 0.88rem;
            font-weight: 500; padding: 8px 14px; border-radius: 999px;
            border: 1px solid transparent;
            transition: color .15s ease, background .15s ease, border-color .15s ease;
        }
        .iris-nav .nav-links a:hover {
            color: var(--text); background: var(--surface-soft);
            border-color: var(--border);
        }
        .iris-nav .nav-links a.active {
            color: var(--accent); background: var(--accent-soft);
            border-color: rgba(37, 99, 235, 0.18);
        }
        .iris-nav .nav-back {
            display: flex; align-items: center; justify-content: center;
            width: 36px; height: 36px;
            color: var(--accent); text-decoration: none; font-size: 1.2rem;
            font-weight: 600; line-height: 1;
            background: var(--surface-soft);
            border: 1px solid var(--border);
            border-radius: 11px;
            transition: background .15s ease, box-shadow .15s ease;
        }
        .iris-nav .nav-back:hover {
            background: #fff;
            box-shadow: var(--shadow-xs);
        }
        .iris-nav .work-name {
            font-size: 1rem; font-weight: 600; letter-spacing: -0.01em;
        }

        /* ---- Workflow / results screens ---- */
        .work-screen {margin-bottom: 0;}
        .work-screen .preview-sticky {
            position: sticky;
            top: 60px;
            z-index: 40;
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(10px);
            border-radius: 0 0 var(--radius-md) var(--radius-md);
            box-shadow: var(--shadow-sm);
        }
        .work-screen.correction-mode .preview-frame {margin-bottom: 0;}
        .work-screen.correction-mode .preview-frame img {
            max-height: 34vh;
            width: auto;
            max-width: 100%;
            margin: 0 auto;
            object-fit: contain;
        }
        .ocu-ipr.compact {
            font-size: 0.95rem;
            padding: 6px 0 8px;
        }

        .iris-analyzing {
            text-align: center;
            padding: 28px 16px 32px;
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text);
            letter-spacing: 0.01em;
        }

        /* ---- Cards ---- */
        .iris-card {
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            background: var(--surface);
            padding: 24px 26px;
            margin-bottom: 18px;
            box-shadow: var(--shadow-md);
        }
        .iris-card h3 {
            font-size: 1rem; font-weight: 700; letter-spacing: -0.02em;
            color: var(--text); margin: 0 0 14px;
        }

        /* ---- Metric tiles ---- */
        .metric-panel {
            display: flex;
            flex-direction: column;
            gap: 12px;
            padding-bottom: 4px;
        }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        @media (min-width: 640px) {
            .metric-row {
                grid-template-columns: repeat(4, 1fr);
            }
        }
        .metric {
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            background: var(--surface-soft);
            padding: 16px;
            box-shadow: var(--shadow-xs);
            height: 84px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
            overflow: hidden;
        }
        .metric .label {
            color: var(--text-dim); font-size: 0.7rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.05em;
            line-height: 1.3;
            flex-shrink: 0;
        }
        .metric .value {
            font-size: 1.45rem; font-weight: 700; color: var(--text);
            letter-spacing: -0.02em;
            font-variant-numeric: tabular-nums;
            line-height: 1.2;
            flex-shrink: 0;
        }
        .metric .value.text {
            font-size: 1.15rem;
        }
        .metric .value.coords {
            font-size: 1.25rem;
        }
        .metric .unit {color: var(--text-dim); font-size: 0.8rem; font-weight: 400;}
        .metric.hl {
            background: linear-gradient(145deg, #eff6ff, #f8fbff);
            border-color: rgba(37, 99, 235, 0.2);
            box-shadow: 0 4px 14px var(--accent-glow);
        }
        .metric.hl .value {color: var(--accent);}

        /* ---- Form controls ---- */
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label {
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
            color: var(--text-dim) !important;
        }
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiSelect"] > div > div {
            background: var(--surface) !important;
            border: 1px solid var(--border-2) !important;
            border-radius: var(--radius-sm) !important;
            box-shadow: var(--shadow-xs) !important;
        }

        /* ---- Buttons ---- */
        .stButton>button {
            border-radius: var(--radius-sm);
            padding: 0.6rem 1.1rem;
            box-shadow: var(--shadow-xs);
            transition: background .15s ease, border-color .15s ease,
                        box-shadow .15s ease, transform .12s ease;
        }
        .stButton>button:active {transform: scale(0.985);}
        .stButton>button,
        .stButton>button *,
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-primary"] *,
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-secondary"] * {
            font-size: 14px !important; font-weight: 600 !important;
            line-height: 1.5 !important; font-family: inherit !important;
        }
        .stButton>button p {margin: 0;}
        .stButton>button[kind="primary"] {
            background: linear-gradient(180deg, #3b82f6 0%, var(--accent) 100%);
            color: #fff; border: 1px solid rgba(29, 78, 216, 0.35);
            box-shadow: 0 2px 8px var(--accent-glow);
        }
        .stButton>button[kind="primary"]:hover,
        .stButton>button[kind="primary"]:focus {
            background: linear-gradient(180deg, var(--accent) 0%, var(--accent-h) 100%);
            border-color: var(--accent-h); color: #fff;
            box-shadow: 0 4px 14px var(--accent-glow);
        }
        .stButton>button[kind="secondary"] {
            background: var(--surface); color: var(--text);
            border: 1px solid var(--border-2);
        }
        .stButton>button[kind="secondary"]:hover,
        .stButton>button[kind="secondary"]:focus {
            background: var(--surface-soft); color: var(--text);
            border-color: var(--border-2);
            box-shadow: var(--shadow-sm);
        }

        .stDownloadButton>button {
            background: var(--surface); color: var(--text); font-weight: 600;
            border: 1px solid var(--border-2); border-radius: var(--radius-sm);
            padding: 0.6rem 1.1rem; box-shadow: var(--shadow-xs);
        }
        .stDownloadButton>button:hover {
            background: var(--surface-soft); border-color: var(--border-2);
            box-shadow: var(--shadow-sm);
        }

        div[role="radiogroup"] {gap: 8px;}

        .nudge-label {font-size: 0.85rem; color: var(--text);}
        .nudge-val {
            text-align: center; font-size: 0.9rem; font-weight: 700;
            color: var(--text); font-variant-numeric: tabular-nums;
        }

        /* ---- Preview & IPR ---- */
        .preview-frame {
            background: linear-gradient(160deg, #e8ecf4 0%, #dde3ee 100%);
            display: flex; align-items: center; justify-content: center;
            margin-bottom: 10px; overflow: hidden;
            padding: 14px;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-2);
            box-shadow: inset 0 1px 4px rgba(15, 23, 42, 0.06), var(--shadow-md);
        }
        .preview-frame img {
            display: block; width: 100%; height: auto;
            border: none !important;
            border-radius: calc(var(--radius-sm) - 2px) !important;
            box-shadow: var(--shadow-sm);
        }

        .ocu-ipr {
            text-align: center;
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text);
            font-variant-numeric: tabular-nums;
            padding: 14px 0 16px;
            letter-spacing: -0.01em;
        }

        /* ---- Manual correction panel ---- */
        .manual-correction-panel {
            background: var(--surface);
            padding: 12px 14px;
            margin: 10px 0;
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }
        .manual-correction-panel.compact {
            padding: 10px 12px;
            margin: 4px 0 8px;
        }
        .manual-correction-panel.compact h4,
        .manual-correction-panel h4 {
            text-align: center;
            font-size: 0.68rem; font-weight: 700;
            margin: 0 0 8px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .correction-select {
            display: flex; flex-direction: row; gap: 8px;
            margin-bottom: 12px;
        }
        .manual-correction-panel.compact .correction-select {
            gap: 6px; margin-bottom: 10px;
        }
        .correction-select .sel-btn {
            flex: 1 1 0; display: block; text-align: center;
            padding: 0.55rem 0.5rem;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-2);
            background: var(--surface-soft);
            color: var(--text-dim);
            text-decoration: none;
            font-size: 13px; font-weight: 600; line-height: 1.4;
            box-shadow: var(--shadow-xs);
            transition: all .15s ease;
        }
        .manual-correction-panel.compact .correction-select .sel-btn {
            padding: 0.35rem 0.4rem; font-size: 11px;
        }
        .correction-select .sel-btn.active {
            background: linear-gradient(180deg, #3b82f6 0%, var(--accent) 100%);
            border-color: rgba(29, 78, 216, 0.4);
            color: #fff;
            box-shadow: 0 3px 10px var(--accent-glow);
        }

        .correction-pads-row {
            display: flex; flex-direction: row; flex-wrap: nowrap;
            align-items: flex-start; gap: 10px; width: 100%;
        }
        .correction-pad {flex: 1 1 0; min-width: 0;}

        .pad-label {
            text-align: center;
            font-size: 0.68rem; font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 6px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .dpad-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 5px;
        }
        .dpad-gap {display: block; min-height: 1px;}

        .pad-btn {
            display: flex; align-items: center; justify-content: center;
            min-height: 36px;
            background: linear-gradient(180deg, #fff 0%, var(--surface-soft) 100%);
            color: var(--accent);
            border: 1px solid var(--border-2);
            border-radius: 9px;
            text-decoration: none;
            font-size: 0.9rem; font-weight: 600;
            box-shadow: var(--shadow-xs);
            -webkit-tap-highlight-color: transparent;
            user-select: none;
            transition: background .12s ease, box-shadow .12s ease, transform .1s ease;
        }
        .pad-btn:hover {
            background: #fff;
            box-shadow: var(--shadow-sm);
            border-color: rgba(37, 99, 235, 0.25);
        }
        .pad-btn:active {
            transform: scale(0.96);
            background: var(--surface-soft);
            box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.08);
        }

        .fine-btns {
            display: flex; flex-direction: column; gap: 5px;
        }
        .fine-btn {
            min-height: 40px; padding: 6px 2px;
            color: var(--accent);
        }
        .fine-btn svg {display: block; margin: 0 auto;}

        @media (max-width: 480px) {
            .pad-label {font-size: 0.6rem;}
            .iris-nav {padding: 0 16px;}
            .iris-nav .tagline {
                font-size: 0.58rem;
                letter-spacing: 0.04em;
            }
        }

        /* Hidden Streamlit buttons that back the HTML correction pads */
        .stApp:has(#correction-adj-hooks) [class*="st-key-adj_"] {
            display: none !important;
            height: 0 !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
        }

        /* file uploader rendered as a single solid button */
        [data-testid="stFileUploader"] > label {display: none;}
        [data-testid="stFileUploader"] {width: 100% !important;}
        [data-testid="stFileUploaderDropzone"] {
            display: flex !important; align-items: center !important;
            justify-content: center !important;
            width: 100% !important; min-height: 0;
            padding: 0.6rem 1.1rem; cursor: pointer;
            background: linear-gradient(180deg, #3b82f6 0%, var(--accent) 100%);
            border: 1px solid rgba(29, 78, 216, 0.35);
            border-radius: var(--radius-sm);
            box-shadow: 0 2px 8px var(--accent-glow);
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            background: linear-gradient(180deg, var(--accent) 0%, var(--accent-h) 100%);
            box-shadow: 0 4px 14px var(--accent-glow);
        }
        [data-testid="stFileUploaderDropzone"] * {display: none !important;}
        [data-testid="stFileUploaderDropzone"]::after {
            content: "Upload image"; color: #fff;
            font-size: 14px !important; font-weight: 600 !important;
            line-height: 1.5 !important; font-family: inherit !important;
            width: 100%; text-align: center;
        }
        [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] ul {display: none !important;}

        /* expander, dataframe, images, alerts */
        [data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            background: var(--surface) !important;
            box-shadow: var(--shadow-xs) !important;
            overflow: visible !important;
        }
        [data-testid="stExpander"] summary {
            font-weight: 600 !important;
            color: var(--text) !important;
        }
        [data-testid="stExpanderDetails"] {
            padding-bottom: 16px !important;
            overflow: visible !important;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-xs);
        }
        [data-testid="stImage"] img {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-sm);
        }
        [data-testid="stAlert"] {
            border-radius: var(--radius-sm) !important;
            border: 1px solid var(--border) !important;
        }

        /* No dim overlay while the app is processing */
        .stApp[data-testscript-state="running"] .stAppViewContainer,
        .stApp[data-testscript-state="running"] .main .block-container {
            opacity: 1 !important;
            filter: none !important;
        }
        [data-testid="stSpinner"],
        [data-testid="stSpinnerOverlay"] {
            display: none !important;
        }

        /* ---- About page ---- */
        .iris-card p {
            color: var(--text-dim); font-size: 0.94rem;
            line-height: 1.65; margin: 0 0 12px;
        }
        .iris-card p:last-child {margin-bottom: 0;}
        .iris-card a {color: var(--accent); text-decoration: none; font-weight: 500;}
        .iris-card a:hover {text-decoration: underline;}
        .about-output {display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap;}
        .about-output .shot {flex: 0 0 200px;}
        .about-output .shot img {
            width: 200px; height: auto;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-md);
        }
        .about-output .legend {flex: 1; min-width: 240px;}
        .about-output .legend ul {list-style: none; padding: 0; margin: 0 0 10px;}
        .about-output .legend li {
            font-size: 0.92rem; margin: 8px 0; color: var(--text-dim);
        }
        .about-output .legend .key {font-size: 1rem; margin-right: 8px;}

        /* ---- Footer ---- */
        .iris-footer {
            display: flex; align-items: center; justify-content: center; gap: 8px;
            margin: 44px 0 16px; padding-top: 22px;
            border-top: 1px solid var(--border);
            color: var(--text-muted); font-size: 0.84rem; font-weight: 500;
        }
        .iris-footer .heart {color: #e11d48; font-size: 0.92rem;}
        .iris-footer a {display: inline-flex; align-items: center; text-decoration: none;}
        .iris-footer img {
            height: 22px; width: auto; vertical-align: middle;
            border: none !important; cursor: pointer;
            opacity: 0.92;
        }
        .iris-footer a:hover img {opacity: 1;}
        .iris-footer .lab {font-weight: 600; color: var(--text);}
        </style>
        """,
        unsafe_allow_html=True,
    )
