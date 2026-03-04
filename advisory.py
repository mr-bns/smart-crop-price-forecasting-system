"""
Advisory recommendation engine — v2.0
Generates farmer-friendly actionable advice in English and Telugu.
"""


ADVISORY_MESSAGES = {
    "en": {
        "rise":   (
            "📈 Prices are expected to increase. "
            "Farmers may consider holding the crop before selling to capture better returns."
        ),
        "fall":   (
            "📉 Prices may decrease soon. "
            "Selling earlier may be more beneficial — consider offloading stock at current market rates."
        ),
        "stable": (
            "📊 Market prices are expected to remain stable. "
            "You have flexibility in your selling decision — monitor daily for any sudden changes."
        ),
    },
    "te": {
        "rise":   (
            "📈 ధరలు పెరిగే అవకాశం ఉంది. "
            "మంచి ఆదాయం పొందడానికి రైతులు పంటను అమ్మే ముందు కొంతకాలం వేచి ఉండవచ్చు."
        ),
        "fall":   (
            "📉 ధరలు త్వరలో తగ్గవచ్చు. "
            "ముందే అమ్మడం మరింత లాభదాయకం — ప్రస్తుత మార్కెట్ ధరలకు నిల్వలు విక్రయించడాన్ని పరిగణించండి."
        ),
        "stable": (
            "📊 మార్కెట్ ధరలు స్థిరంగా ఉంటాయని ఆశించబడుతోంది. "
            "మీ అమ్మకం నిర్ణయంలో మీకు సౌలభ్యం ఉంది — ఏదైనా ఆకస్మిక మార్పుల కోసం రోజువారీ పర్యవేక్షించండి."
        ),
    }
}


def generate_advisory(low, high, forecast, lang_dict, lang_code: str = "en") -> tuple:
    """
    Generate a sell/hold/wait advisory based on forecast vs current market.

    Args:
        low:       market low price
        high:      market high price
        forecast:  predicted price
        lang_dict: LANG dict for current language (kept for backward compatibility)
        lang_code: 'en' or 'te' (default 'en')

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
