"""
app.py
------
FLAGGERR — Invoice Intelligence Platform
By Ansh Mishra
Run:  streamlit run app.py
"""

import sys
from pathlib import Path
import streamlit as st
import joblib
import numpy as np

ROOT        = Path(__file__).parent
FREIGHT_DIR = ROOT / "freight_prediction_project"
INVOICE_DIR = ROOT / "invoice_of_freight_classification"

sys.path.insert(0, str(FREIGHT_DIR))
sys.path.insert(0, str(INVOICE_DIR))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FLAGGERR",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ── Apple-style ease curve everywhere ── */
:root {
    --ease: cubic-bezier(0.4, 0, 0.2, 1);
    --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
    --blue: #2997ff;
    --green: #30d158;
    --red: #ff453a;
}

.stApp {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, #0d1a2b 0%, #000000 55%);
    color: #f5f5f7;
}

/* subtle animated aurora backdrop, Apple keynote style */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(circle at 15% 20%, rgba(41,151,255,0.08), transparent 40%),
        radial-gradient(circle at 85% 10%, rgba(48,209,88,0.06), transparent 35%);
    animation: drift 18s ease-in-out infinite alternate;
}
@keyframes drift {
    0%   { transform: translate(0,0) scale(1); }
    100% { transform: translate(2%, 3%) scale(1.05); }
}

[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #1d1d1f;
}

/* ── Fade-up entrance for main content ── */
[data-testid="stAppViewContainer"] .main .block-container {
    animation: fadeUp 0.7s var(--ease-out) both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── NAV BRAND ── */
.brand {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: #f5f5f7;
    padding: 4px 0 2px 0;
    transition: letter-spacing 0.3s var(--ease);
}
.brand-sub {
    font-size: 0.72rem;
    color: #6e6e73;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 28px;
}

/* ── HERO ── */
.hero-wrap {
    text-align: center;
    padding: 80px 24px 56px 24px;
    animation: fadeUp 0.9s var(--ease-out) both;
}
.hero-ship {
    font-size: 4.5rem;
    display: block;
    margin-bottom: 20px;
    filter: drop-shadow(0 0 32px rgba(100,180,255,0.35));
    animation: float 4s ease-in-out infinite, bob 4s ease-in-out infinite;
}
@keyframes float {
    0%,100% { transform: translateY(0px) rotate(0deg); }
    50%      { transform: translateY(-10px) rotate(-2deg); }
}
@keyframes bob { 0%,100% { filter: drop-shadow(0 0 32px rgba(100,180,255,0.35)); } 50% { filter: drop-shadow(0 0 46px rgba(100,180,255,0.55)); } }

.hero-title {
    font-size: clamp(2.8rem, 6vw, 5.4rem);
    font-weight: 800;
    letter-spacing: -2.5px;
    line-height: 1.05;
    color: #f5f5f7;
    margin-bottom: 16px;
    background: linear-gradient(180deg, #ffffff 0%, #cfcfd4 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-title .blue {
    background: linear-gradient(90deg, #2997ff, #5ac8fa 50%, #30d158);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-desc {
    font-size: 1.15rem;
    color: #86868b;
    font-weight: 400;
    max-width: 560px;
    margin: 0 auto 40px auto;
    line-height: 1.6;
}
.hero-divider {
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, #2997ff, #30d158);
    margin: 0 auto 48px auto;
    border-radius: 2px;
}

/* ── STAT CHIPS ── */
.chip-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 0;
}
.chip {
    background: #1c1c1e;
    border: 1px solid #2c2c2e;
    border-radius: 100px;
    padding: 8px 18px;
    font-size: 0.82rem;
    color: #aeaeb2;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: transform 0.25s var(--ease), border-color 0.25s var(--ease), background 0.25s var(--ease);
}
.chip:hover {
    transform: translateY(-3px);
    border-color: #2997ff80;
    background: #202022;
}
.chip b { color: #f5f5f7; }

/* ── SECTION LABEL ── */
.section-eye {
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #2997ff;
    font-weight: 600;
    margin-bottom: 8px;
}
.section-title {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.8px;
    color: #f5f5f7;
    margin-bottom: 6px;
}
.section-sub {
    font-size: 0.92rem;
    color: #6e6e73;
    margin-bottom: 32px;
    line-height: 1.5;
}

/* ── GLASS CARD ── */
.glass {
    background: rgba(28,28,30,0.85);
    border: 1px solid #2c2c2e;
    border-radius: 18px;
    padding: 32px;
    backdrop-filter: blur(20px);
    margin-bottom: 20px;
    transition: border-color 0.3s var(--ease), transform 0.3s var(--ease), box-shadow 0.3s var(--ease);
}
.glass:hover {
    border-color: #3a3a3c;
    box-shadow: 0 20px 40px -20px rgba(0,0,0,0.6);
}

/* ── METRIC TILES ── */
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 32px; }
.tile {
    background: #1c1c1e;
    border: 1px solid #2c2c2e;
    border-radius: 14px;
    padding: 20px 24px;
    flex: 1;
    min-width: 120px;
    transition: transform 0.25s var(--ease), border-color 0.25s var(--ease);
}
.tile:hover {
    transform: translateY(-2px);
    border-color: #2997ff60;
}
.tile-label {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6e6e73;
    margin-bottom: 6px;
}
.tile-val {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f5f5f7;
    letter-spacing: -1px;
    line-height: 1;
}
.tile-val .unit { font-size: 1rem; color: #6e6e73; font-weight: 400; }

/* ── INPUTS ── */
label { color: #aeaeb2 !important; font-size: 0.85rem !important; font-weight: 500 !important; }
.stNumberInput input {
    background: #1c1c1e !important;
    border: 1px solid #3a3a3c !important;
    border-radius: 10px !important;
    color: #f5f5f7 !important;
    font-size: 1rem !important;
    transition: border-color 0.25s var(--ease), box-shadow 0.25s var(--ease) !important;
}
.stNumberInput input:focus {
    border-color: #2997ff !important;
    box-shadow: 0 0 0 4px rgba(41,151,255,0.15) !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(180deg, #34a1ff, #2997ff);
    color: #fff;
    border: none;
    border-radius: 980px;
    padding: 13px 36px;
    font-size: 0.95rem;
    font-weight: 600;
    width: 100%;
    letter-spacing: -0.2px;
    transition: all 0.25s var(--ease);
    cursor: pointer;
    box-shadow: 0 4px 18px -6px rgba(41,151,255,0.55);
}
.stButton > button:hover {
    background: linear-gradient(180deg, #4dabff, #0077ed);
    transform: scale(1.02) translateY(-1px);
    box-shadow: 0 8px 26px -6px rgba(41,151,255,0.75);
}
.stButton > button:active {
    transform: scale(0.98);
}

/* ── RESULT CARDS ── */
.res-normal, .res-flag, .res-freight {
    animation: fadeUp 0.5s var(--ease-out) both;
    transition: transform 0.3s var(--ease);
}
.res-normal:hover, .res-flag:hover, .res-freight:hover { transform: translateY(-3px); }

.res-normal {
    background: linear-gradient(135deg, #0a1f0a, #0d2b10);
    border: 1px solid #30d15840;
    border-radius: 18px;
    padding: 36px;
    text-align: center;
}
.res-flag {
    background: linear-gradient(135deg, #1f0a0a, #2b0d0d);
    border: 1px solid #ff453a40;
    border-radius: 18px;
    padding: 36px;
    text-align: center;
}
.res-freight {
    background: linear-gradient(135deg, #0a0f1f, #0d152b);
    border: 1px solid #2997ff40;
    border-radius: 18px;
    padding: 36px;
    text-align: center;
}
.res-eyebrow {
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6e6e73;
    margin-bottom: 10px;
}
.res-number {
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -2px;
    color: #f5f5f7;
    line-height: 1;
    margin-bottom: 8px;
}
.res-note { font-size: 0.88rem; color: #6e6e73; }

/* ── PROGRESS BAR ── */
.prob-wrap { margin-bottom: 16px; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #6e6e73;
    margin-bottom: 6px;
}
.prob-label b { color: #f5f5f7; }
.prob-track {
    background: #2c2c2e;
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
}
.prob-fill-green { background: #30d158; height: 6px; border-radius: 100px; transition: width 0.8s var(--ease-out); }
.prob-fill-red   { background: #ff453a; height: 6px; border-radius: 100px; transition: width 0.8s var(--ease-out); }

/* ── WHY BOX ── */
.why-box {
    background: #1c1c1e;
    border: 1px solid #2c2c2e;
    border-radius: 14px;
    padding: 20px 24px;
    margin-top: 16px;
    font-size: 0.88rem;
    color: #aeaeb2;
    line-height: 1.6;
    animation: fadeUp 0.6s var(--ease-out) both;
    transition: border-color 0.25s var(--ease);
}
.why-box:hover { border-color: #3a3a3c; }
.why-box b { color: #f5f5f7; }
.why-box .hi { color: #2997ff; font-weight: 600; }

/* ── EMPTY STATE ── */
.empty {
    background: #1c1c1e;
    border: 1px dashed #3a3a3c;
    border-radius: 18px;
    padding: 60px 32px;
    text-align: center;
    color: #3a3a3c;
    transition: border-color 0.3s var(--ease);
}
.empty:hover { border-color: #2997ff60; }
.empty .icon {
    font-size: 2.5rem;
    margin-bottom: 12px;
    display: inline-block;
    animation: sway 3s ease-in-out infinite;
}
@keyframes sway {
    0%,100% { transform: rotate(-4deg); }
    50%      { transform: rotate(4deg); }
}
.empty p { font-size: 0.9rem; }

/* ── SIDEBAR NAV ── */
.stRadio > div { gap: 4px !important; }
.stRadio > div > label {
    background: transparent !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #aeaeb2 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    transition: all 0.2s var(--ease) !important;
}
.stRadio > div > label:hover {
    background: #1c1c1e !important;
    transform: translateX(3px);
}

hr { border-color: #1d1d1f !important; margin: 20px 0 !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── CREDIT FOOTER ── */
.credit-footer {
    text-align: center;
    padding: 40px 24px 24px 24px;
    color: #3a3a3c;
    font-size: 0.8rem;
    letter-spacing: 0.3px;
}
.credit-footer .wave { display: inline-block; animation: sway 3s ease-in-out infinite; }
.credit-footer b { color: #6e6e73; font-weight: 600; }

.sidebar-credit {
    font-size: 0.76rem;
    color: #3a3a3c;
    line-height: 1.8;
    padding-top: 10px;
}
.sidebar-credit b { color: #6e6e73; }
</style>
""", unsafe_allow_html=True)


# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_freight_model():
    m = joblib.load(FREIGHT_DIR / "saved_models" / "predict_freight_model.pkl")
    s = joblib.load(FREIGHT_DIR / "saved_models" / "scaler.pkl")
    return m, s

@st.cache_resource
def load_invoice_model():
    m = joblib.load(INVOICE_DIR / "models" / "predict_flag_invoice.pkl")
    s = joblib.load(INVOICE_DIR / "models" / "scaler.pkl")
    return m, s


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class='brand'>🚢 FLAGGERR</div>
        <div class='brand-sub'>Invoice Intelligence</div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "Freight Cost Predictor",
        "Invoice Risk Detector"
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-size:0.78rem;color:#3a3a3c;line-height:2;'>
            <div style='color:#6e6e73;font-weight:600;letter-spacing:2px;font-size:0.7rem;
                        text-transform:uppercase;margin-bottom:8px;'>Stack</div>
            Python · scikit-learn<br>
            pandas · joblib<br>
            SQLite · Streamlit
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        <div class='sidebar-credit'>
            Designed &amp; built by<br><b>Ansh Mishra</b>
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FREIGHT COST PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "Freight Cost Predictor":

    st.markdown("""
    <div class='hero-wrap'>
        <span class='hero-ship'>🚢</span>
        <div class='hero-title'>Predict your<br><span class='blue'>Freight Cost</span></div>
        <div class='hero-desc'>
            Enter the invoice dollar amount. Our Linear Regression model
            predicts the exact freight cost — trained on real vendor invoice data.
        </div>
        <div class='hero-divider'></div>
        <div class='chip-row'>
            <div class='chip'>⚡ <b>Linear Regression</b></div>
            <div class='chip'>📈 R² <b>96.99%</b></div>
            <div class='chip'>📉 MAE <b>24.11</b></div>
            <div class='chip'>🎯 RMSE <b>124.72</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("""
            <div class='section-eye'>Input</div>
            <div class='section-title'>Invoice Details</div>
            <div class='section-sub'>Enter the total dollar value of the vendor invoice below.</div>
        """, unsafe_allow_html=True)

        dollars = st.number_input(
            "Invoice Amount (USD)",
            min_value=0.0, max_value=500000.0,
            value=5000.0, step=100.0,
            help="Total dollar value of the vendor invoice"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Calculate Freight Cost →")

    with col_out:
        st.markdown("""
            <div class='section-eye'>Result</div>
            <div class='section-title'>Prediction</div>
            <div class='section-sub'>Your predicted freight cost will appear here.</div>
        """, unsafe_allow_html=True)

        if predict_btn:
            try:
                model, scaler = load_freight_model()
                scaled        = scaler.transform(np.array([[dollars]]))
                pred          = model.predict(scaled)[0]
                ratio         = (pred / dollars * 100) if dollars > 0 else 0

                st.markdown(f"""
                <div class='res-freight'>
                    <div class='res-eyebrow'>Predicted Freight Cost</div>
                    <div class='res-number'>${pred:,.2f}</div>
                    <div class='res-note'>for a ${dollars:,.2f} invoice</div>
                </div>
                <div class='why-box'>
                    Freight ratio: <span class='hi'>{ratio:.2f}%</span> of invoice value —
                    <b>${pred:,.2f}</b> goes toward freight on this shipment.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Run `main.py` first to generate saved_models/")
        else:
            st.markdown("""
            <div class='empty'>
                <div class='icon'>🚢</div>
                <p>Your prediction will appear here</p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INVOICE RISK DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
else:

    st.markdown("""
    <div class='hero-wrap'>
        <span class='hero-ship'>🚨</span>
        <div class='hero-title'>Invoice Risk<br><span class='blue'>Detector</span></div>
        <div class='hero-desc'>
            FLAGGERR scans vendor invoices for abnormal cost discrepancies
            and delivery delays — flagging those that need manual review.
        </div>
        <div class='hero-divider'></div>
        <div class='chip-row'>
            <div class='chip'>🌲 <b>Random Forest</b></div>
            <div class='chip'>🎯 Accuracy <b>89%</b></div>
            <div class='chip'>🔍 Precision <b>96%</b></div>
            <div class='chip'>📡 Recall <b>71%</b></div>
            <div class='chip'>⚙️ GridSearchCV <b>tuned</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section-eye'>Input</div>
        <div class='section-title'>Invoice Details</div>
        <div class='section-sub'>Fill in all fields from the vendor invoice to check its risk level.</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        invoice_quantity    = st.number_input("Invoice Quantity",       min_value=0,   value=100,    step=1)
        invoice_dollars     = st.number_input("Invoice Dollars ($)",    min_value=0.0, value=5000.0, step=100.0)
    with col2:
        freight             = st.number_input("Freight Cost ($)",       min_value=0.0, value=150.0,  step=10.0)
        days_po_to_invoice  = st.number_input("Days  PO → Invoice",     min_value=0,   value=5,      step=1)
    with col3:
        total_item_quantity = st.number_input("Total Item Quantity",    min_value=0,   value=100,    step=1)
        total_item_dollars  = st.number_input("Total Item Dollars ($)", min_value=0.0, value=4990.0, step=100.0)

    st.markdown("<br>", unsafe_allow_html=True)
    flag_btn = st.button("Analyse Invoice Risk →")

    if flag_btn:
        try:
            model, scaler = load_invoice_model()
            inp    = np.array([[invoice_quantity, invoice_dollars, freight,
                                days_po_to_invoice, total_item_quantity, total_item_dollars]])
            scaled = scaler.transform(inp)
            pred   = model.predict(scaled)[0]
            proba  = model.predict_proba(scaled)[0]
            disc   = abs(invoice_dollars - total_item_dollars)

            st.markdown("<br>", unsafe_allow_html=True)
            col_res, col_detail = st.columns([1, 1], gap="large")

            with col_res:
                if pred == 1:
                    st.markdown(f"""
                    <div class='res-flag'>
                        <div class='res-eyebrow'>Risk Assessment</div>
                        <div class='res-number'>⚠️ FLAGGED</div>
                        <div class='res-note'>This invoice needs manual review</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='res-normal'>
                        <div class='res-eyebrow'>Risk Assessment</div>
                        <div class='res-number'>✅ NORMAL</div>
                        <div class='res-note'>No anomalies detected</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='why-box'>
                    Dollar discrepancy: <span class='hi'>${disc:,.2f}</span>
                    {"  ⚠️ exceeds $5 threshold" if disc > 5 else "  ✅ within $5 threshold"}
                </div>
                """, unsafe_allow_html=True)

            with col_detail:
                st.markdown(f"""
                <div class='glass'>
                    <div class='section-eye'>Confidence</div>
                    <div style='margin-top:20px;'>
                        <div class='prob-wrap'>
                            <div class='prob-label'>
                                <span>✅ Normal</span>
                                <b>{proba[0]*100:.1f}%</b>
                            </div>
                            <div class='prob-track'>
                                <div class='prob-fill-green' style='width:{proba[0]*100:.1f}%'></div>
                            </div>
                        </div>
                        <div class='prob-wrap'>
                            <div class='prob-label'>
                                <span>⚠️ Flagged</span>
                                <b>{proba[1]*100:.1f}%</b>
                            </div>
                            <div class='prob-track'>
                                <div class='prob-fill-red' style='width:{proba[1]*100:.1f}%'></div>
                            </div>
                        </div>
                    </div>
                    <div style='margin-top:24px;border-top:1px solid #2c2c2e;padding-top:20px;'>
                        <div class='tiles'>
                            <div class='tile'>
                                <div class='tile-label'>Discrepancy</div>
                                <div class='tile-val'>${disc:,.0f}</div>
                            </div>
                            <div class='tile'>
                                <div class='tile-label'>Confidence</div>
                                <div class='tile-val'>{max(proba)*100:.0f}<span class='unit'>%</span></div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Run `train.py` first to generate models/")


# ── Footer credit ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='credit-footer'>
    <span class='wave'>🌊</span>&nbsp; FLAGGERR — Invoice Intelligence Platform &nbsp;·&nbsp; Built by <b>Ansh Mishra</b>
</div>
""", unsafe_allow_html=True)