def generate_advisory(low, high, forecast, lang):
    """
    Generate a sell/hold/wait advisory based on forecast vs current market.
    Returns (message_string, key) where key is 'rise', 'fall', or 'stable'.
    """
    if low is None or high is None:
        return lang["advisory"]["stable"], "stable"

    avg = (low + high) / 2
    if avg == 0:
        return lang["advisory"]["stable"], "stable"

    diff_pct = (forecast - avg) / avg * 100

    if diff_pct > 7:
        key = "rise"       # forecast significantly above market → wait to sell
    elif diff_pct < -7:
        key = "fall"       # forecast below market → sell now
    else:
        key = "stable"     # within ±7% → market stable

    return lang["advisory"][key], key
