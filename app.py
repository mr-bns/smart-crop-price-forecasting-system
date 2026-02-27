import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import json
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    from model.predict import run_prediction
    from market_data import get_market_price
    from advisory import generate_advisory
    from language_map import LANG
    from ui.components import (
        crop_image_html, reliability_badge, confidence_bar,
        metric_card, advisory_card, section_header, footer_html,
    )
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/crops.csv"
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

CROPS    = ["Tomato","Onion","Potato","Paddy","BitterGourd","Brinjal","BroadBeans","Carrot","GreenChilli","Okra"]
DISTRICTS = ["Krishna","Guntur","Visakhapatnam","EastWestGodavari","Chittoor","Kurnool"]

UI_TEXT = {
    "en": {
        "lang_label":    "Language",
        "badge":         "AI-POWERED · ANDHRA PRADESH · CROP MARKETS",
        "title_line1":   "Smart Crop Price",
        "title_line2":   "Forecasting System",
        "sub":           "AI-Driven Market Intelligence for Data-Backed Farming Decisions.",
        "start_btn":     "▼  Start Forecasting",
        "how_tag":       "HOW IT WORKS",
        "how_title":     "From Data to Decision in Seconds",
        "how_sub":       "Our three-stage AI pipeline delivers reliable price intelligence",
        "card1_step":    "Step 01",  "card1_title": "Data Collection",
        "card1_body":    "Historical mandi prices across 6 AP districts curated daily. Seasonal patterns, festival cycles, and harvest calendars encoded into structured time-series.",
        "card2_step":    "Step 02",  "card2_title": "AI Forecasting",
        "card2_body":    "XGBoost models — one per crop-district pair — learn lag prices, rolling trends, weekly/yearly seasonality, and harvest calendar signals for accurate short-term forecasts.",
        "card3_step":    "Step 03",  "card3_title": "Advisory & Insights",
        "card3_body":    "Predictions validated with real RMSE & MAPE on held-out test data. A Reliability score guides farmers with actionable sell / hold decisions.",
        "form_tag":      "FORECAST ENGINE",
        "form_title":    "Generate Your Price Forecast",
        "form_sub":      "Select crop, district, and target date",
        "form_heading":  "🔍 Configure Prediction",
        "form_desc":     "Fill in all fields to run the AI model",
        "crop_label":    "🌱 Select Crop",
        "dist_label":    "📍 Select District",
        "date_label":    "📅 Target Date",
        "predict_btn":   "🚀 Generate Forecast",
        "res_tag":       "FORECAST RESULTS",
        "trend_tag":     "TREND & ANALYTICS",
        "trend_title":   "Price Trend & Forecast Overlay",
        "trend_sub":     "30-day historical prices with AI forecast and confidence interval",
        "dl_btn":        "📥  Download Forecast Report",
        "spinning":      "🧠 AI model is computing your forecast…",
        "no_model":      "No trained model found. Please run: python -m model.train",
        "past_date":     "The selected date is too far in the past.",
        "fc_limit":      "Forecast is limited to 30 days ahead.",
        "generic_err":   "An unexpected error occurred. Please try again.",
        "stat_models":   "Trained Models", "stat_crops": "Crops",
        "stat_dist":     "Districts",      "stat_window": "Forecast Window",
        "pred_label":    "Predicted Price", "mkt_label": "Market Range",
        "vs_label":      "vs Current Market", "conf_label": "Confidence",
        "rmse_label":    "RMSE",
        "pred_sub":      "per kg · XGBoost AI",
        "mkt_sub":       "live mandi price",
        "vs_sub":        "forecast vs mandi avg",
        "conf_sub":      "model confidence",
        "rmse_sub":      "root mean squared error",
        "live_range":    "Live Market Range",
        "trained_lbl":   "Model Last Trained",
        "pts_lbl":       "Training Data Points",
        "algo_lbl":      "Algorithm",
        "algo_val":      "XGBoost Regressor",
        "season_lbl":    "Seasonality",
        "season_val":    "Yearly · Monthly · Weekly · Harvest",
        "rel_lbl":       "Reliability Rating",
    },
    "te": {
        "lang_label":    "భాష",
        "badge":         "AI ఆధారిత · ఆంధ్ర ప్రదేశ్ · పంట మార్కెట్లు",
        "title_line1":   "స్మార్ట్ పంట ధర",
        "title_line2":   "అంచనా వ్యవస్థ",
        "sub":           "డేటా ఆధారిత వ్యవసాయ నిర్ణయాల కోసం AI మార్కెట్ ఇంటెలిజెన్స్.",
        "start_btn":     "▼  అంచనా ప్రారంభించండి",
        "how_tag":       "ఇది ఎలా పనిచేస్తుంది",
        "how_title":     "డేటా నుండి నిర్ణయానికి సెకన్లలో",
        "how_sub":       "మా మూడు-దశల AI పైప్‌లైన్ విశ్వసనీయ ధర అంచనాలు అందిస్తుంది",
        "card1_step":    "దశ 01",   "card1_title": "డేటా సేకరణ",
        "card1_body":    "6 ఆంధ్రప్రదేశ్ జిల్లాల మండి ధరలు రోజువారీ సేకరించబడతాయి. పంట కాలాలు మరియు ఉత్సవ చక్రాలు ఎన్కోడ్ చేయబడ్డాయి.",
        "card2_step":    "దశ 02",   "card2_title": "AI అంచనా",
        "card2_body":    "XGBoost మోడళ్లు — ప్రతి పంట-జిల్లా జంటకు ఒకటి — lag ధరలు, rolling trends, మరియు పంట కాల సంకేతాలను నేర్చుకుంటాయి.",
        "card3_step":    "దశ 03",   "card3_title": "సలహా & అంతర్దృష్టులు",
        "card3_body":    "అంచనాలు నిజమైన RMSE & MAPE తో ధృవీకరించబడతాయి. విశ్వసనీయత స్కోర్ రైతులకు అమ్మడం/వేచి ఉండడం నిర్ణయించడంలో సహాయపడుతుంది.",
        "form_tag":      "అంచనా ఇంజిన్",
        "form_title":    "మీ ధర అంచనా రూపొందించండి",
        "form_sub":      "పంట, జిల్లా మరియు తేదీ ఎంచుకోండి",
        "form_heading":  "🔍 అంచనా కాన్ఫిగర్ చేయండి",
        "form_desc":     "AI మోడల్ అమలు చేయడానికి అన్ని ఫీల్డ్‌లు నింపండి",
        "crop_label":    "🌱 పంట ఎంచుకోండి",
        "dist_label":    "📍 జిల్లా ఎంచుకోండి",
        "date_label":    "📅 అంచనా తేదీ",
        "predict_btn":   "🚀 అంచనా రూపొందించండి",
        "res_tag":       "అంచనా ఫలితాలు",
        "trend_tag":     "ధరల ట్రెండ్ & విశ్లేషణ",
        "trend_title":   "ధర ట్రెండ్ & అంచనా ఓవర్‌లే",
        "trend_sub":     "AI అంచనాతో 30-రోజుల చారిత్రక ధరలు",
        "dl_btn":        "📥  అంచనా నివేదిక డౌన్‌లోడ్ చేయండి",
        "spinning":      "🧠 AI మోడల్ మీ అంచనాను లెక్కిస్తోంది…",
        "no_model":      "శిక్షణ పొందిన మోడల్ కనుగొనబడలేదు. దయచేసి అమలు చేయండి: python -m model.train",
        "past_date":     "ఎంచుకున్న తేదీ చాలా గతంలో ఉంది.",
        "fc_limit":      "అంచనా 30 రోజులకు మాత్రమే పరిమితం.",
        "generic_err":   "అనుకోని లోపం సంభవించింది. మళ్ళీ ప్రయత్నించండి.",
        "stat_models":   "శిక్షణ పొందిన మోడళ్లు", "stat_crops": "పంటలు",
        "stat_dist":     "జిల్లాలు",                "stat_window": "అంచనా విండో",
        "pred_label":    "ఊహించిన ధర",  "mkt_label": "మార్కెట్ పరిధి",
        "vs_label":      "ప్రస్తుత మార్కెట్‌తో",   "conf_label": "నమ్మకం",
        "rmse_label":    "RMSE",
        "pred_sub":      "కేజీకి · XGBoost AI",
        "mkt_sub":       "లైవ్ మండి ధర",
        "vs_sub":        "అంచనా vs మండి సగటు",
        "conf_sub":      "మోడల్ నమ్మకం",
        "rmse_sub":      "root mean squared error",
        "live_range":    "లైవ్ మార్కెట్ పరిధి",
        "trained_lbl":   "మోడల్ చివరగా శిక్షణ పొందింది",
        "pts_lbl":       "శిక్షణ డేటా పాయింట్లు",
        "algo_lbl":      "అల్గారిథమ్",
        "algo_val":      "XGBoost Regressor",
        "season_lbl":    "సీజనాలిటీ",
        "season_val":    "వార్షిక · మాసిక · వారపు · పంట",
        "rel_lbl":       "విశ్వసనీయత రేటింగ్",
    }
}

CROP_DISPLAY_EN = {
    "Tomato":"Tomato","Onion":"Onion","Potato":"Potato","Paddy":"Paddy",
    "BitterGourd":"Bitter Gourd","Brinjal":"Brinjal","BroadBeans":"Broad Beans",
    "Carrot":"Carrot","GreenChilli":"Green Chilli","Okra":"Okra",
}
DISTRICT_DISPLAY = {
    "Krishna":"Krishna","Guntur":"Guntur","Visakhapatnam":"Visakhapatnam",
    "EastWestGodavari":"East & West Godavari","Chittoor":"Chittoor","Kurnool":"Kurnool",
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Crop Price Forecasting System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("lang","en"),("result",None),("show_res",False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE SHORTCUT
# ─────────────────────────────────────────────────────────────────────────────
T  = UI_TEXT[st.session_state.lang]
TL = LANG.get(st.session_state.lang, LANG["en"])

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800;900&display=swap');

html { scroll-behavior: smooth; }
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #0a1a0c; min-height: 100vh; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Section base ── */
.section {
    min-height: 85vh;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2.5rem 2rem;
    position: relative;
}
.section-compact { min-height: auto; padding: 2rem 2rem; }

/* ── HERO ── */
.hero-section {
    background: linear-gradient(135deg, #081508 0%, #0d2e10 40%, #1a4a1e 70%, #0f3012 100%);
    overflow: hidden;
    min-height: 90vh;
}
.hero-section::before {
    content:''; position:absolute; top:-50%; left:-50%; width:200%; height:200%;
    background: radial-gradient(ellipse at 30% 30%, rgba(67,160,71,0.15) 0%, transparent 55%),
                radial-gradient(ellipse at 75% 75%, rgba(27,94,32,0.18) 0%, transparent 55%);
    animation: aurora 9s ease-in-out infinite alternate; pointer-events:none;
}
@keyframes aurora {
    0%   { transform: translate(0,0) rotate(0deg); }
    100% { transform: translate(-3%,-3%) rotate(3deg); }
}
.hero-section::after {
    content:''; position:absolute; bottom:0; left:0; right:0; height:90px;
    background: linear-gradient(to bottom, transparent, #0a1a0c); pointer-events:none;
}
.hero-content { position:relative; z-index:2; text-align:center; max-width:820px; }

.hero-badge {
    display:inline-flex; align-items:center; gap:0.5rem;
    background: rgba(67,160,71,0.12); border: 1px solid rgba(67,160,71,0.35);
    border-radius:50px; padding:0.35rem 1.1rem; font-size:0.75rem;
    font-weight:700; color:#81c784; letter-spacing:0.1em; text-transform:uppercase;
    margin-bottom:1.2rem; backdrop-filter:blur(10px);
}
.badge-dot {
    width:6px; height:6px; background:#66bb6a; border-radius:50%;
    animation: pulse-dot 1.8s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:0.5;transform:scale(1.5);} }

.hero-title {
    font-family:'Poppins',sans-serif; font-size:clamp(2rem,4.5vw,3.6rem);
    font-weight:900; color:#fff; line-height:1.15; margin:0 0 1rem; letter-spacing:-0.5px;
}
.hero-title span {
    background: linear-gradient(135deg, #66bb6a, #43a047, #a5d6a7);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-sub {
    font-size:clamp(0.9rem,1.8vw,1.1rem); color:rgba(255,255,255,0.6);
    line-height:1.65; margin:0 0 1.8rem; max-width:580px; margin-left:auto; margin-right:auto;
}
.hero-pills {
    display:flex; flex-wrap:wrap; gap:0.5rem; justify-content:center; margin-bottom:2rem;
}
.hero-pill {
    background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1);
    border-radius:20px; padding:0.28rem 0.85rem; font-size:0.78rem; color:rgba(255,255,255,0.55);
}
.scroll-btn {
    display:inline-flex; align-items:center; gap:0.5rem;
    background:linear-gradient(135deg,#2e7d32,#43a047); color:#fff;
    border:none; border-radius:50px; padding:0.8rem 2rem;
    font-size:0.95rem; font-weight:700; cursor:pointer; text-decoration:none;
    box-shadow:0 8px 28px rgba(46,125,50,0.4); transition:all 0.3s ease;
    animation: float-btn 3s ease-in-out infinite;
}
.scroll-btn:hover { transform:translateY(-3px) scale(1.04); box-shadow:0 12px 36px rgba(46,125,50,0.5); color:#fff; text-decoration:none; animation:none; }
@keyframes float-btn { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-5px);} }

.hero-stats { display:flex; gap:2.5rem; justify-content:center; margin-top:2.5rem; border-top:1px solid rgba(255,255,255,0.07); padding-top:1.8rem; flex-wrap:wrap; }
.stat-item { text-align:center; }
.stat-number { font-family:'Poppins',sans-serif; font-size:1.9rem; font-weight:800; color:#66bb6a; line-height:1; }
.stat-label  { font-size:0.75rem; color:rgba(255,255,255,0.4); margin-top:0.25rem; }

/* ── HOW IT WORKS ── */
.how-section { background:linear-gradient(170deg,#0a1a0c 0%,#0d2010 100%); }
.section-tag { display:inline-block; background:rgba(67,160,71,0.1); border:1px solid rgba(67,160,71,0.28); border-radius:30px; padding:0.3rem 0.9rem; font-size:0.72rem; font-weight:700; color:#81c784; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.8rem; }
.section-title { font-family:'Poppins',sans-serif; font-size:clamp(1.6rem,3vw,2.2rem); font-weight:800; color:#fff; margin-bottom:0.4rem; text-align:center; }
.section-sub { color:rgba(255,255,255,0.4); font-size:0.95rem; text-align:center; margin-bottom:2.5rem; }

.how-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:1.2rem; max-width:960px; width:100%; }
.how-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:18px; padding:1.8rem 1.5rem; transition:all 0.32s cubic-bezier(0.4,0,0.2,1); }
.how-card:hover { transform:translateY(-7px) scale(1.02); background:rgba(67,160,71,0.07); border-color:rgba(67,160,71,0.28); box-shadow:0 18px 50px rgba(67,160,71,0.13); }
.how-icon { font-size:2.2rem; margin-bottom:0.8rem; display:block; }
.how-step { font-size:0.68rem; font-weight:700; color:#43a047; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.4rem; }
.how-card h3 { font-family:'Poppins',sans-serif; font-size:1.1rem; font-weight:700; color:#fff; margin:0 0 0.5rem; }
.how-card p { color:rgba(255,255,255,0.45); font-size:0.87rem; line-height:1.6; margin:0; }

/* ── INPUT ── */
.input-section { background:linear-gradient(160deg,#0d2010 0%,#0a1a0c 60%,#0d2010 100%); }
.glass-card { background:rgba(255,255,255,0.035); border:1px solid rgba(255,255,255,0.09); border-radius:22px; padding:2.2rem 2rem; max-width:620px; width:100%; backdrop-filter:blur(20px); box-shadow:0 20px 70px rgba(0,0,0,0.38); }
.glass-card h2 { font-family:'Poppins',sans-serif; font-size:1.4rem; font-weight:700; color:#fff; margin:0 0 0.3rem; }
.glass-card p { color:rgba(255,255,255,0.4); font-size:0.87rem; margin-bottom:1.6rem; }

/* ── Widget dark overrides ── */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background:rgba(255,255,255,0.05) !important; border:1.5px solid rgba(255,255,255,0.1) !important;
    border-radius:11px !important; color:#e0e0e0 !important;
}
div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"] > div:focus-within {
    border-color:#43a047 !important; box-shadow:0 0 0 3px rgba(67,160,71,0.18) !important;
}
div[data-baseweb="select"] svg { color:#81c784 !important; }
div[data-baseweb="popover"] { background:#182e1a !important; }
div[data-baseweb="option"]:hover { background:rgba(67,160,71,0.14) !important; }
label[data-testid="stWidgetLabel"] > div > p { color:rgba(255,255,255,0.65) !important; font-weight:500 !important; }

div.stButton > button {
    background:linear-gradient(135deg,#2e7d32,#43a047) !important; color:#fff !important;
    border:none !important; border-radius:11px !important; padding:0.7rem 1.6rem !important;
    font-weight:700 !important; font-size:0.97rem !important; letter-spacing:0.02em !important;
    transition:all 0.22s ease !important; box-shadow:0 5px 20px rgba(46,125,50,0.32) !important;
}
div.stButton > button:hover { transform:translateY(-2px) scale(1.05) !important; box-shadow:0 9px 28px rgba(46,125,50,0.45) !important; }
div.stButton > button:active { transform:translateY(0) scale(0.98) !important; }

/* ── RESULTS ── */
.results-section { background:linear-gradient(170deg,#0a1a0c 0%,#081208 60%,#0a1a0c 100%); padding:2rem 2rem; min-height:auto; }
.results-header-card { background:linear-gradient(135deg,rgba(27,94,32,0.55),rgba(46,125,50,0.35)); border:1px solid rgba(67,160,71,0.28); border-radius:18px; padding:1.5rem 1.8rem; margin-bottom:1.5rem; }
.results-crop-name { font-family:'Poppins',sans-serif; font-size:1.8rem; font-weight:800; color:#fff; margin:0 0 0.25rem; }
.results-meta { color:rgba(255,255,255,0.55); font-size:0.88rem; }

/* 5-col metric grid */
.metric-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:0.85rem; margin-bottom:1.2rem; }
@media(max-width:900px){ .metric-grid{ grid-template-columns:repeat(3,1fr); } }
@media(max-width:600px){ .metric-grid{ grid-template-columns:repeat(2,1fr); } }

.metric-card { background:rgba(255,255,255,0.035); border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:1.1rem 0.9rem; transition:all 0.28s ease; text-align:center; }
.metric-card:hover { background:rgba(67,160,71,0.07); border-color:rgba(67,160,71,0.22); box-shadow:0 7px 26px rgba(67,160,71,0.11); transform:translateY(-3px); }
.metric-label { font-size:0.68rem; font-weight:700; color:rgba(255,255,255,0.38); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem; }
.metric-value { font-family:'Poppins',sans-serif; font-size:1.6rem; font-weight:800; color:#fff; line-height:1; margin-bottom:0.25rem; }
.metric-sub { font-size:0.7rem; color:rgba(255,255,255,0.3); }

/* Badges */
.badge { display:inline-flex; align-items:center; gap:0.35rem; border-radius:30px; padding:0.28rem 0.85rem; font-size:0.78rem; font-weight:700; letter-spacing:0.04em; }
.badge-high   { background:rgba(102,187,106,0.12); border:1px solid rgba(102,187,106,0.35); color:#81c784; }
.badge-medium { background:rgba(255,202,40,0.1);   border:1px solid rgba(255,202,40,0.3);   color:#ffca28; }
.badge-low    { background:rgba(239,83,80,0.1);    border:1px solid rgba(239,83,80,0.3);    color:#ef5350; }

/* Confidence bar */
.conf-section { margin:1.2rem 0; }
.conf-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem; }
.conf-label { font-size:0.78rem; color:rgba(255,255,255,0.45); font-weight:600; }
.conf-value { font-size:0.88rem; font-weight:700; color:#fff; }
.conf-bar-bg { background:rgba(255,255,255,0.07); border-radius:7px; height:9px; overflow:hidden; }
.conf-bar-fill { height:9px; border-radius:7px; animation:growBar 1.4s cubic-bezier(0.4,0,0.2,1) forwards; }
.conf-bar-high   { background:linear-gradient(90deg,#2e7d32,#66bb6a); }
.conf-bar-medium { background:linear-gradient(90deg,#e65100,#ffca28); }
.conf-bar-low    { background:linear-gradient(90deg,#b71c1c,#ef5350); }
@keyframes growBar { from{width:0%;} }

/* Advisory */
.advisory-card { border-radius:14px; padding:1.1rem 1.3rem; margin:1.2rem 0; border-left-width:4px; border-left-style:solid; font-size:0.92rem; line-height:1.6; font-weight:500; }
.advisory-rise   { background:rgba(25,118,210,0.08); border-left-color:#1976d2; color:#90caf9; }
.advisory-fall   { background:rgba(245,127,23,0.08); border-left-color:#f57f17; color:#ffcc80; }
.advisory-stable { background:rgba(67,160,71,0.08);  border-left-color:#43a047; color:#a5d6a7; }

/* ── TREND ── */
.trend-section { background:linear-gradient(160deg,#081208 0%,#0a1a0c 100%); min-height:auto; padding:2rem 2rem; }
.meta-row { display:flex; flex-wrap:wrap; gap:1.2rem; margin-top:1.2rem; padding:1rem 1.3rem; background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.06); border-radius:12px; }
.meta-item { display:flex; flex-direction:column; }
.meta-key { font-size:0.68rem; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.18rem; }
.meta-val { font-size:0.9rem; color:#e0e0e0; font-weight:600; }

/* ── Fade-in animation ── */
.fade-in { opacity:0; transform:translateY(24px); animation:fadeInUp 0.65s ease forwards; }
@keyframes fadeInUp { to{opacity:1;transform:translateY(0);} }
.fade-in-1{animation-delay:0.08s;} .fade-in-2{animation-delay:0.2s;} .fade-in-3{animation-delay:0.32s;} .fade-in-4{animation-delay:0.48s;}

/* ── Language toggle ── */
.lang-pos { position:absolute; top:1.2rem; right:1.5rem; z-index:10; }

/* ── Footer ── */
.footer-bar { background:#050d06; text-align:center; padding:1.5rem 2rem; color:rgba(255,255,255,0.18); font-size:0.78rem; border-top:1px solid rgba(255,255,255,0.04); }

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;} ::-webkit-scrollbar-track{background:#0a1a0c;} ::-webkit-scrollbar-thumb{background:#2e7d32;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA CACHE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()

@st.cache_resource
def cached_model(crop, region):
    """Cache loaded models in memory to avoid repeated disk reads."""
    import joblib
    path = os.path.join("models", f"{crop}_{region}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HERO  (with language toggle)
# ─────────────────────────────────────────────────────────────────────────────
hero_left, hero_right = st.columns([5, 1])
with hero_right:
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    lang_choice = st.selectbox(
        T["lang_label"], ["English", "Telugu (తెలుగు)"],
        index=0 if st.session_state.lang == "en" else 1,
        label_visibility="collapsed", key="lang_sel"
    )
    new_lang = "en" if lang_choice == "English" else "te"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

st.markdown(f"""
<div class="section hero-section" id="hero">
  <div class="hero-content">
    <div class="hero-badge"><span class="badge-dot"></span>{T['badge']}</div>
    <h1 class="hero-title">
      {T['title_line1']}<br><span>{T['title_line2']}</span>
    </h1>
    <p class="hero-sub">{T['sub']}</p>
    <div class="hero-pills">
      <span class="hero-pill">🍅 Tomato</span>
      <span class="hero-pill">🧅 Onion</span>
      <span class="hero-pill">🌾 Paddy</span>
      <span class="hero-pill">🌶️ Chilli</span>
      <span class="hero-pill">🥕 Carrot</span>
      <span class="hero-pill">+ 5 {('more' if st.session_state.lang=='en' else 'మరిన్ని')}</span>
    </div>
    <a href="#how" class="scroll-btn">{T['start_btn']}</a>
    <div class="hero-stats">
      <div class="stat-item"><div class="stat-number">60</div><div class="stat-label">{T['stat_models']}</div></div>
      <div class="stat-item"><div class="stat-number">10</div><div class="stat-label">{T['stat_crops']}</div></div>
      <div class="stat-item"><div class="stat-number">6</div><div class="stat-label">{T['stat_dist']}</div></div>
      <div class="stat-item"><div class="stat-number">30d</div><div class="stat-label">{T['stat_window']}</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="section how-section" id="how">
  <div class="section-tag">{T['how_tag']}</div>
  <div class="section-title">{T['how_title']}</div>
  <p class="section-sub">{T['how_sub']}</p>
  <div class="how-cards">
    <div class="how-card fade-in fade-in-1">
      <span class="how-icon">📡</span>
      <div class="how-step">{T['card1_step']}</div>
      <h3>{T['card1_title']}</h3>
      <p>{T['card1_body']}</p>
    </div>
    <div class="how-card fade-in fade-in-2">
      <span class="how-icon">🤖</span>
      <div class="how-step">{T['card2_step']}</div>
      <h3>{T['card2_title']}</h3>
      <p>{T['card2_body']}</p>
    </div>
    <div class="how-card fade-in fade-in-3">
      <span class="how-icon">💡</span>
      <div class="how-step">{T['card3_step']}</div>
      <h3>{T['card3_title']}</h3>
      <p>{T['card3_body']}</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FORECAST INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="section input-section section-compact" id="forecast">
  <div class="section-tag">{T['form_tag']}</div>
  <div class="section-title">{T['form_title']}</div>
  <p class="section-sub">{T['form_sub']}</p>
""", unsafe_allow_html=True)

_, col_form, _ = st.columns([1, 2, 1])
with col_form:
    st.markdown(f'<div class="glass-card"><h2>{T["form_heading"]}</h2><p>{T["form_desc"]}</p>', unsafe_allow_html=True)

    crop_names = [TL["crops"].get(c, c) for c in CROPS]
    sel_crop_disp = st.selectbox(T["crop_label"], crop_names, key="crop_sel")
    crop_key = CROPS[crop_names.index(sel_crop_disp)]

    dist_names = [TL["regions"].get(d, DISTRICT_DISPLAY[d]) for d in DISTRICTS]
    sel_dist_disp = st.selectbox(T["dist_label"], dist_names, key="dist_sel")
    region_key = DISTRICTS[dist_names.index(sel_dist_disp)]

    default_date = datetime.today() + timedelta(days=1)
    selected_date = st.date_input(
        T["date_label"],
        value=default_date,
        min_value=datetime.today(),
        max_value=datetime.today() + timedelta(days=30),
        key="date_sel"
    )

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    predict_clicked = st.button(T["predict_btn"], use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if predict_clicked:
    st.session_state.show_res = False
    with st.spinner(T["spinning"]):
        time.sleep(1.2)
        try:
            result = run_prediction(crop_key, region_key, datetime.combine(selected_date, datetime.min.time()))
            low, high = get_market_price(region_key, crop_key)
            result["market_low"]  = low
            result["market_high"] = high
            result["crop_key"]    = crop_key
            result["region_key"]  = region_key
            result["date"]        = selected_date
            st.session_state.result   = result
            st.session_state.show_res = True
        except ValueError as e:
            st.session_state.result   = {"error": str(e)}
            st.session_state.show_res = True

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.show_res and st.session_state.result:
    res  = st.session_state.result
    st.markdown(f'<div class="section results-section section-compact" id="results">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-tag">{T["res_tag"]}</div>', unsafe_allow_html=True)

    if "error" in res:
        err_map = {
            "NO_DATA":        T["no_model"],
            "PAST_DATE":      T["past_date"],
            "FORECAST_LIMIT": T["fc_limit"],
            "GENERIC_ERROR":  T["generic_err"],
        }
        st.error(f"⚠️ {err_map.get(res['error'], res['error'])}")
    else:
        ck           = res["crop_key"]
        rk           = res["region_key"]
        price        = res["price"]
        confidence   = res["confidence"]
        rmse         = res["rmse"]
        reliability  = res["reliability"]
        low          = res.get("market_low")
        high         = res.get("market_high")
        date_str     = res["date"].strftime("%d %b %Y") if hasattr(res["date"], "strftime") else str(res["date"])
        crop_disp    = TL["crops"].get(ck, ck)
        region_disp  = TL["regions"].get(rk, DISTRICT_DISPLAY.get(rk, rk))

        # Header with crop image
        img_html = crop_image_html(ck, size=80)
        badge_html = reliability_badge(reliability)
        st.markdown(f"""
        <div class="results-header-card fade-in">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
            <div style="display:flex;align-items:center;gap:1rem;">
              {img_html}
              <div>
                <div class="results-crop-name">{crop_disp}</div>
                <div class="results-meta">📍 {region_disp} &nbsp;|&nbsp; 📅 {date_str}</div>
              </div>
            </div>
            <div>{badge_html}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 5 Metric cards (NO MAPE)
        market_avg  = round((low + high) / 2, 2) if low and high else None
        diff        = round(price - market_avg, 2) if market_avg else None
        diff_str    = (("+" if diff >= 0 else "") + f"₹{diff}") if diff is not None else "N/A"
        diff_color  = "#66bb6a" if diff and diff >= 0 else "#ef5350"
        conf_color  = "#66bb6a" if confidence >= 90 else ("#ffca28" if confidence >= 80 else "#ef5350")
        mkt_str     = f"₹{round(low,1)}–₹{round(high,1)}" if low and high else "N/A"
        rmse_str    = f"₹{round(rmse,2)}" if rmse is not None else "—"

        st.markdown(f"""
        <div class="metric-grid fade-in fade-in-1">
          {metric_card(T["pred_label"], f"₹{price}", T["pred_sub"], "#66bb6a")}
          {metric_card(T["mkt_label"],  mkt_str,     T["mkt_sub"])}
          {metric_card(T["vs_label"],   diff_str,    T["vs_sub"], diff_color)}
          {metric_card(T["conf_label"], f"{confidence}%", T["conf_sub"], conf_color)}
          {metric_card(T["rmse_label"], rmse_str,    T["rmse_sub"])}
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown(confidence_bar(confidence), unsafe_allow_html=True)

        # Advisory
        if low and high:
            advice, advice_key = generate_advisory(low, high, price, TL)
        else:
            advice, advice_key = TL["advisory"]["stable"], "stable"
        st.markdown(advisory_card(advice, advice_key), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TREND & ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.show_res and st.session_state.result and "error" not in st.session_state.result:
    res    = st.session_state.result
    df_all = load_data()
    ck     = res["crop_key"]
    rk     = res["region_key"]
    price  = res["price"]

    st.markdown(f'<div class="section trend-section section-compact" id="trends">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-tag">{T['trend_tag']}</div>
    <div class="section-title">{T['trend_title']}</div>
    <p class="section-sub">{T['trend_sub']}</p>
    """, unsafe_allow_html=True)

    if not df_all.empty:
        trend_df = df_all[(df_all["crop"] == ck) & (df_all["region"] == rk)].sort_values("date").tail(30)

        if not trend_df.empty:
            hist_dates  = list(trend_df["date"])
            hist_prices = list(trend_df["price"].round(2))
            last_date   = hist_dates[-1]
            tgt_ts      = pd.Timestamp(res["date"])
            fc_days     = max(1, (tgt_ts - last_date).days)

            fc_dates  = [last_date + pd.Timedelta(days=i) for i in range(fc_days + 1)]
            fc_prices = [round(hist_prices[-1] + (price - hist_prices[-1]) * (i / max(fc_days,1)), 2) for i in range(fc_days+1)]

            price_std = float(trend_df["price"].std())
            band = price_std * 1.2
            fc_upper = [round(p + band, 2) for p in fc_prices]
            fc_lower = [round(max(1, p - band), 2) for p in fc_prices]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fc_dates + fc_dates[::-1], y=fc_upper + fc_lower[::-1],
                fill="toself", fillcolor="rgba(67,160,71,0.07)",
                line=dict(color="rgba(0,0,0,0)"), name="Confidence Band", hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=hist_dates, y=hist_prices, mode="lines+markers",
                name="Historical", line=dict(color="#43a047", width=2.5),
                marker=dict(size=4, color="#66bb6a"),
                hovertemplate="<b>%{x|%d %b}</b><br>₹%{y}/kg<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=fc_dates, y=fc_prices, mode="lines+markers",
                name="AI Forecast", line=dict(color="#ffca28", width=2.5, dash="dot"),
                marker=dict(size=5, color="#ffca28", symbol="diamond"),
                hovertemplate="<b>%{x|%d %b}</b><br>Forecast: ₹%{y}/kg<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[fc_dates[-1]], y=[price], mode="markers+text",
                marker=dict(size=11, color="#ffca28", symbol="star"),
                text=[f"₹{price}"], textposition="top center",
                textfont=dict(color="#fff", size=11), name="Predicted Price",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="rgba(255,255,255,0.6)", size=11),
                legend=dict(bgcolor="rgba(255,255,255,0.03)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)", tickprefix="₹"),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="rgba(8,18,8,0.9)", bordercolor="rgba(67,160,71,0.35)", font=dict(color="#fff", size=11)),
                margin=dict(l=0, r=0, t=8, b=8), height=390,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Model metadata
    rel_icon = {"High":"🟢","Medium":"🟡","Low":"🔴"}.get(res.get("reliability","Medium"),"🟡")
    st.markdown(f"""
    <div class="meta-row fade-in">
      <div class="meta-item"><span class="meta-key">{T['trained_lbl']}</span><span class="meta-val">📅 {res.get('trained_date','N/A')}</span></div>
      <div class="meta-item"><span class="meta-key">{T['pts_lbl']}</span><span class="meta-val">📊 {res.get('n_points','N/A')} records</span></div>
      <div class="meta-item"><span class="meta-key">{T['algo_lbl']}</span><span class="meta-val">🤖 {T['algo_val']}</span></div>
      <div class="meta-item"><span class="meta-key">{T['season_lbl']}</span><span class="meta-val">🔄 {T['season_val']}</span></div>
      <div class="meta-item"><span class="meta-key">{T['rel_lbl']}</span><span class="meta-val">{rel_icon} {res.get('reliability','N/A')}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Download Report
    r = res
    crop_disp   = TL["crops"].get(r["crop_key"], r["crop_key"])
    region_disp = TL["regions"].get(r["region_key"], DISTRICT_DISPLAY.get(r["region_key"], r["region_key"]))
    date_str    = r["date"].strftime("%d %b %Y") if hasattr(r["date"],"strftime") else str(r["date"])

    report_html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Crop Price Forecast Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:700px;margin:40px auto;color:#222;}}
h1{{color:#1b5e20;}}h2{{color:#2e7d32;border-bottom:2px solid #a5d6a7;padding-bottom:6px;}}
table{{width:100%;border-collapse:collapse;margin:10px 0;}}
th{{background:#1b5e20;color:white;padding:8px 12px;text-align:left;}}
td{{padding:8px 12px;border-bottom:1px solid #e0e0e0;}}
tr:nth-child(even){{background:#f9f9f9;}}
</style></head><body>
<h1>🌿 Smart Crop Price Forecasting System</h1>
<p><em>AI-Driven Market Intelligence for Data-Backed Farming Decisions</em></p>
<h2>Forecast Summary</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Crop</td><td>{crop_disp}</td></tr>
  <tr><td>District</td><td>{region_disp}</td></tr>
  <tr><td>Target Date</td><td>{date_str}</td></tr>
  <tr><td>Predicted Price</td><td>₹{r['price']} per kg</td></tr>
  <tr><td>Market Range</td><td>{'₹'+str(round(r['market_low'],1))+' – ₹'+str(round(r['market_high'],1)) if r.get('market_low') else 'N/A'}</td></tr>
  <tr><td>Confidence</td><td>{r['confidence']}%</td></tr>
  <tr><td>RMSE</td><td>{'₹'+str(r['rmse']) if r.get('rmse') else '—'}</td></tr>
  <tr><td>Reliability</td><td>{r.get('reliability','N/A')}</td></tr>
  <tr><td>Algorithm</td><td>XGBoost Regressor</td></tr>
  <tr><td>Model Trained</td><td>{r.get('trained_date','N/A')}</td></tr>
  <tr><td>Data Points Used</td><td>{r.get('n_points','N/A')}</td></tr>
</table>
<p style="font-size:11px;color:#888;margin-top:30px;">
  Smart Crop Price Forecasting System · XGBoost ML Engine · Andhra Pradesh Markets<br>
  Report Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}
</p></body></html>"""

    b64   = base64.b64encode(report_html.encode()).decode()
    fname = f"forecast_{r['crop_key']}_{r['region_key']}_{r.get('date', datetime.today()).strftime('%Y%m%d') if hasattr(r.get('date',''), 'strftime') else 'report'}.html"

    _, dl_col, _ = st.columns([1, 1, 1])
    with dl_col:
        st.markdown(f"""
        <a href="data:text/html;base64,{b64}" download="{fname}"
           style="display:block;text-align:center;background:linear-gradient(135deg,#2e7d32,#43a047);
                  color:white;border-radius:11px;padding:0.7rem 1.5rem;font-weight:700;
                  text-decoration:none;font-size:0.92rem;margin-top:1.2rem;
                  box-shadow:0 5px 20px rgba(46,125,50,0.32);">
          {T['dl_btn']}
        </a>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(footer_html(), unsafe_allow_html=True)
st.markdown("""
<script>
document.querySelectorAll('a[href^="#"]').forEach(a=>{
  a.addEventListener('click',e=>{
    e.preventDefault();
    const t=document.querySelector(a.getAttribute('href'));
    if(t) t.scrollIntoView({behavior:'smooth',block:'start'});
  });
});
const obs=new IntersectionObserver((entries)=>{
  entries.forEach(e=>{
    if(e.isIntersecting){
      e.target.style.opacity='1';
      e.target.style.transform='translateY(0)';
      obs.unobserve(e.target);
    }
  });
},{threshold:0.1});
document.querySelectorAll('.fade-in').forEach(el=>obs.observe(el));
</script>
""", unsafe_allow_html=True)
