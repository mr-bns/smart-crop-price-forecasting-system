"""
Advisory recommendation engine — v2.1
Generates farmer-friendly actionable advice and rule-based explanations.
"""

ADVISORY_MESSAGES = {
    "en": {
        "rise": (
            "📈 Prices are expected to increase. "
            "Farmers may consider holding the crop before selling to capture better returns."
        ),
        "fall": (
            "📉 Prices may decrease soon. "
            "Selling earlier may be more beneficial — consider offloading stock at current market rates."
        ),
        "stable": (
            "📊 Market prices are expected to remain stable. "
            "You have flexibility in your selling decision — monitor daily for any sudden changes."
        ),
    },
    "te": {
        "rise": (
            "📈 ధరలు పెరిగే అవకాశం ఉంది. "
            "మంచి ఆదాయం పొందడానికి రైతులు పంటను అమ్మే ముందు కొంతకాలం వేచి ఉండవచ్చు."
        ),
        "fall": (
            "📉 ధరలు త్వరలో తగ్గవచ్చు. "
            "ముందే అమ్మడం మరింత లాభదాయకం — ప్రస్తుత మార్కెట్ ధరలకు నిల్వలు విక్రయించడాన్ని పరిగణించండి."
        ),
        "stable": (
            "📊 మార్కెట్ ధరలు స్థిరంగా ఉంటాయని ఆశించబడుతోంది. "
            "మీ అమ్మకం నిర్ణయంలో మీకు సౌలభ్యం ఉంది — ఏదైనా ఆకస్మిక మార్పుల కోసం రోజువారీ పర్యవేక్షించండి."
        ),
    }
}

# Season display names for explanation bullets
SEASON_NOTES = {
    "en": {
        "Kharif": "Monsoon season (Kharif) — moderate supply with weather uncertainty.",
        "Rabi":   "Winter season (Rabi) — generally stable supply conditions.",
        "Summer": "Summer season — reduced supply, prices often rise.",
    },
    "te": {
        "Kharif": "వర్షాకాల సీజన్ (ఖరీఫ్) — వాతావరణ అనిశ్చితతో మధ్యస్థ సరఫరా.",
        "Rabi":   "శీతాకాల సీజన్ (రాబీ) — సాధారణంగా స్థిరమైన సరఫరా పరిస్థితులు.",
        "Summer": "వేసవి కాలం — తక్కువ సరఫరా, ధరలు తరచుగా పెరుగుతాయి.",
    }
}

HARVEST_NOTES = {
    "en": {
        True:  "This crop is currently in peak harvest season — high supply tends to lower prices.",
        False: "This crop is outside its peak harvest — tighter supply often pushes prices up.",
    },
    "te": {
        True:  "ఈ పంట ప్రస్తుతం గరిష్ఠ కోత సీజన్‌లో ఉంది — అధిక సరఫరా ధరలను తగ్గిస్తుంది.",
        False: "ఈ పంట గరిష్ఠ కోత కాలంలో లేదు — తక్కువ సరఫరా ధరలను పెంచవచ్చు.",
    }
}

TREND_NOTES = {
    "en": {
        "up":     "Historical prices show an upward trend over the last 30 days.",
        "down":   "Historical prices show a downward trend over the last 30 days.",
        "stable": "Historical prices have been relatively stable over the last 30 days.",
    },
    "te": {
        "up":     "చారిత్రక ధరలు గత 30 రోజులలో పెరిగే ధోరణి చూపాయి.",
        "down":   "చారిత్రక ధరలు గత 30 రోజులలో తగ్గే ధోరణి చూపాయి.",
        "stable": "చారిత్రక ధరలు గత 30 రోజులలో సాపేక్షంగా స్థిరంగా ఉన్నాయి.",
    }
}

MKT_NOTES = {
    "en": {
        "above": "Forecast (₹{pred}) is ABOVE market average (₹{mkt}) — strong demand signal.",
        "below": "Forecast (₹{pred}) is BELOW market average (₹{mkt}) — consider selling now.",
        "near":  "Forecast is close to market average (₹{mkt}) — stable market expected.",
    },
    "te": {
        "above": "అంచనా ధర (₹{pred}) మార్కెట్ సగటు (₹{mkt}) కంటే ఎక్కువ — డిమాండ్ అధికంగా ఉంది.",
        "below": "అంచనా ధర (₹{pred}) మార్కెట్ సగటు (₹{mkt}) కంటే తక్కువ — ఇప్పుడు అమ్మడం మంచిది.",
        "near":  "అంచనా ధర మార్కెట్ సగటు (₹{mkt}) దగ్గరగా ఉంది — మార్కెట్ స్థిరంగా ఉంటుంది.",
    }
}


def generate_advisory(low, high, forecast, lang_dict, lang_code: str = "en") -> tuple:
    """
    Generate a sell/hold/wait advisory based on forecast vs current market.

    Returns:
        (message_string, key)  where key is 'rise', 'fall', or 'stable'.
    """
    if low is None or high is None:
        key = "stable"
        msg = ADVISORY_MESSAGES.get(lang_code, ADVISORY_MESSAGES["en"])[key]
        return msg, key

    avg = (low + high) / 2
    if avg == 0:
        key = "stable"
        msg = ADVISORY_MESSAGES.get(lang_code, ADVISORY_MESSAGES["en"])[key]
        return msg, key

    diff_pct = (forecast - avg) / avg * 100

    if diff_pct > 7:
        key = "rise"
    elif diff_pct < -7:
        key = "fall"
    else:
        key = "stable"

    msg = ADVISORY_MESSAGES.get(lang_code, ADVISORY_MESSAGES["en"])[key]
    return msg, key


def get_explanation_bullets(
    season: str,
    harvest_on: bool,
    trend_slope: float,
    predicted_price: float,
    market_low=None,
    market_high=None,
    lang: str = "en",
) -> list:
    """
    Return a list of 3-4 human-readable explanation bullet strings.
    All logic is rule-based — no ML required.
    """
    lang = lang if lang in ("en", "te") else "en"
    bullets = []

    # 1. Season note
    season_note = SEASON_NOTES.get(lang, SEASON_NOTES["en"]).get(season, f"Current season: {season}")
    bullets.append(season_note)

    # 2. Harvest note
    harvest_note = HARVEST_NOTES.get(lang, HARVEST_NOTES["en"]).get(harvest_on, "")
    if harvest_note:
        bullets.append(harvest_note)

    # 3. Trend note
    if abs(trend_slope) < 0.5:
        trend_key = "stable"
    elif trend_slope > 0:
        trend_key = "up"
    else:
        trend_key = "down"
    bullets.append(TREND_NOTES.get(lang, TREND_NOTES["en"])[trend_key])

    # 4. Market comparison
    if market_low is not None and market_high is not None and market_low > 0 and market_high > 0:
        mkt_avg = round((market_low + market_high) / 2, 2)
        if mkt_avg > 0:
            diff_pct = (predicted_price - mkt_avg) / mkt_avg * 100
            if diff_pct > 7:
                k = "above"
            elif diff_pct < -7:
                k = "below"
            else:
                k = "near"
            tmpl = MKT_NOTES.get(lang, MKT_NOTES["en"])[k]
            bullets.append(tmpl.format(pred=round(predicted_price, 2), mkt=mkt_avg))

    return bullets
