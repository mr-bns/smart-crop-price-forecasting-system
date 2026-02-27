"""
Reusable UI component functions for the Smart Crop Price Forecasting System.
Returns HTML strings rendered via st.markdown(..., unsafe_allow_html=True).
"""

import os
import base64

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

CROP_ASSET = {
    "Tomato":      "tomato.jpg",
    "Onion":       "onion.jpg",
    "Potato":      "potato.jpg",
    "Paddy":       "paddy.jpg",
    "BitterGourd": "bitter_gourd.jpg",
    "Brinjal":     "brinjal.jpg",
    "BroadBeans":  "broad_beans.jpg",
    "Carrot":      "carrot.jpg",
    "GreenChilli": "green_chilli.jpg",
    "Okra":        "okra.jpg",
}


def get_crop_image_b64(crop_key: str) -> str | None:
    """Return base64-encoded image string for inline embedding, or None."""
    fname = CROP_ASSET.get(crop_key)
    if not fname:
        return None
    path = os.path.join(ASSETS_DIR, fname)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def crop_image_html(crop_key: str, size: int = 90) -> str:
    """Return an <img> tag with base64 embedded image, or a fallback icon."""
    b64 = get_crop_image_b64(crop_key)
    if b64:
        ext = CROP_ASSET.get(crop_key, "x.jpg").split(".")[-1]
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        return (
            f'<img src="data:{mime};base64,{b64}" '
            f'style="width:{size}px;height:{size}px;object-fit:cover;'
            f'border-radius:14px;box-shadow:0 4px 16px rgba(0,0,0,0.35);" />'
        )
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:14px;'
        f'background:rgba(255,255,255,0.08);display:flex;align-items:center;'
        f'justify-content:center;font-size:{size//2}px;">🌿</div>'
    )


def reliability_badge(reliability: str) -> str:
    styles = {
        "High":   ("badge-high",   "🟢 High Reliability"),
        "Medium": ("badge-medium", "🟡 Medium Reliability"),
        "Low":    ("badge-low",    "🔴 Low Reliability"),
    }
    cls, label = styles.get(reliability, ("badge-medium", f"🟡 {reliability}"))
    return f'<span class="badge {cls}">{label}</span>'


def confidence_bar(confidence: float) -> str:
    if confidence >= 90:
        bar_cls = "conf-bar-high"
    elif confidence >= 80:
        bar_cls = "conf-bar-medium"
    else:
        bar_cls = "conf-bar-low"

    return f"""
    <div class="conf-section">
      <div class="conf-header">
        <span class="conf-label">Prediction Confidence</span>
        <span class="conf-value">{confidence}%</span>
      </div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill {bar_cls}" style="width:{min(confidence,100)}%"></div>
      </div>
    </div>"""


def metric_card(label: str, value: str, sub: str = "", color: str = "white") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color};">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>"""


def advisory_card(advice: str, key: str) -> str:
    cls_map = {
        "rise":   ("advisory-rise",   "📈 Price Rising —"),
        "fall":   ("advisory-fall",   "📉 Price Falling —"),
        "stable": ("advisory-stable", "✅ Market Stable —"),
    }
    cls, icon = cls_map.get(key, ("advisory-stable", "💡"))
    return f'<div class="advisory-card {cls}"><strong>{icon}</strong> {advice}</div>'


def section_header(tag: str, title: str, subtitle: str = "") -> str:
    sub_html = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    return f"""
    <div class="section-tag">{tag}</div>
    <div class="section-title">{title}</div>
    {sub_html}"""


def footer_html() -> str:
    return """
    <div class="footer-bar">
      🌿 Smart Crop Price Forecasting System &nbsp;•&nbsp;
      AI-Driven Market Intelligence &nbsp;•&nbsp;
      Andhra Pradesh Agricultural Markets &nbsp;•&nbsp;
      XGBoost ML Engine
    </div>"""
