LANG = {

    "en": {
        "title": "🌾 Smart Crop Price Forecasting System",
        "select_lang": "Select your language",
        "select_crop": "Select Crop",
        "select_region": "Select Region",
        "select_date": "Forecast Date",
        "get_advisory": "Get Advisory",

        "live": "Live Market Range",
        "forecast": "AI Forecast",
        "advisory_heading": "Farmer Advisory",

        "system": {
            "FORECAST_LIMIT": "Forecast within 30 days only.",
            "PAST_DATE": "Date is in the past.",
            "NO_DATA": "Historical data unavailable.",
            "NO_MODEL": "No trained model found. Please run: python model/train.py",
            "GENERIC_ERROR": "An unexpected error occurred. Please try again.",
        },

        "advisory": {
            "rise": "Price likely to increase — consider waiting before selling.",
            "fall": "Price may decline — selling soon is recommended.",
            "stable": "Market is stable — flexible decision.",
        },

        "crops": {
            "Tomato":      "Tomato",
            "Onion":       "Onion",
            "Potato":      "Potato",
            "Paddy":       "Paddy",
            "BitterGourd": "Bitter Gourd",
            "Brinjal":     "Brinjal",
            "BroadBeans":  "Broad Beans",
            "Carrot":      "Carrot",
            "GreenChilli": "Green Chilli",
            "Okra":        "Okra (Ladies Finger)",
        },

        "regions": {
            "Krishna":            "Krishna (Vijayawada)",
            "Guntur":             "Guntur",
            "Visakhapatnam":      "Visakhapatnam",
            "EastWestGodavari":   "East/West Godavari",
            "Chittoor":           "Chittoor (Madanapalle Belt)",
            "Kurnool":            "Kurnool",
        },

        "results": {
            "analysis_suffix":    " Analysis",
            "predicted_price":    "Predicted Price",
            "confidence":         "AI Confidence",
            "trend":              "Price Trend (Last 30 Days)",
            "new_search":         "New Search",
            "home":               "Home",
            "loading":            "AI analyzing...",
            "market_unavailable": "Market data unavailable",
            "no_trend":           "No trend data available.",
            "error_prefix":       "Analysis Error",
            "unit_kg":            "/ kg",
        },

        # Extra keys used by refactored app.py
        "data_warning":    "⚠️ Training data is {days} day(s) old — predictions may be less accurate. Run python generate_data.py to refresh.",
        "low_data_warning":"⚠️ Only {n} data points found for this crop/district. Minimum 30 required for reliable prediction.",
        "explanation_title": "Why This Price?",
        "image_alt":        "{crop} crop image",
    },

    "te": {
        "title": "🌾 స్మార్ట్ పంట ధర అంచనా వ్యవస్థ",
        "select_lang": "మీ భాషను ఎంచుకోండి",
        "select_crop": "పంట ఎంచుకోండి",
        "select_region": "ప్రాంతాన్ని ఎంచుకోండి",
        "select_date": "అంచనా తేదీ",
        "get_advisory": "సలహా పొందండి",

        "live": "ప్రస్తుత మార్కెట్ ధరలు",
        "forecast": "కృత్రిమ మేధస్సు అంచనా",
        "advisory_heading": "రైతు సలహా",

        "system": {
            "FORECAST_LIMIT": "అంచనా 30 రోజులు మాత్రమే.",
            "PAST_DATE": "తేదీ గతంలో ఉంది.",
            "NO_DATA": "డేటా లేదు.",
            "NO_MODEL": "శిక్షణ పొందిన మోడల్ లేదు. దయచేసి అమలు చేయండి: python model/train.py",
            "GENERIC_ERROR": "తెలియని లోపం సంభవించింది.",
        },

        "advisory": {
            "rise": "ధరలు పెరిగే అవకాశం — వేచి ఉన్న తర్వాత అమ్మండి.",
            "fall": "ధరలు తగ్గవచ్చు — ఇప్పుడు అమ్మండి.",
            "stable": "మార్కెట్ స్థిరంగా ఉంది — మీ నిర్ణయం తీసుకోండి.",
        },

        "crops": {
            "Tomato":      "టమోటా (Tomato)",
            "Onion":       "ఉల్లిపాయ (Onion)",
            "Potato":      "బంగాళదుంప (Potato)",
            "Paddy":       "వరి (Paddy)",
            "BitterGourd": "కాకరకాయ (Bitter Gourd)",
            "Brinjal":     "వంకాయ (Brinjal)",
            "BroadBeans":  "చిక్కుడు (Broad Beans)",
            "Carrot":      "క్యారెట్ (Carrot)",
            "GreenChilli": "పచ్చి మిరపకాయ (Green Chilli)",
            "Okra":        "బెండకాయ (Okra)",
        },

        "regions": {
            "Krishna":          "కృష్ణా (విజయవాడ)",
            "Guntur":           "గుంటూరు",
            "Visakhapatnam":    "విశాఖపట్నం",
            "EastWestGodavari": "తూర్పు/పశ్చిమ గోదావరి",
            "Chittoor":         "చిత్తూరు (మదనపల్లె)",
            "Kurnool":          "కర్నూలు",
        },

        "results": {
            "analysis_suffix":    " విశ్లేషణ",
            "predicted_price":    "ఊహించిన ధర",
            "confidence":         "AI నమ్మకం",
            "trend":              "ధరల సరళి (గత 30 రోజులు)",
            "new_search":         "కొత్త శోధన",
            "home":               "హోమ్",
            "loading":            "AI విశ్లేషిస్తోంది...",
            "market_unavailable": "మార్కెట్ డేటా అందుబాటులో లేదు",
            "no_trend":           "ధరల సరళి డేటా లేదు",
            "error_prefix":       "విశ్లేషణ లోపం",
            "unit_kg":            "/ కేజీ",
        },

        "data_warning":    "⚠️ శిక్షణ డేటా {days} రోజులు పాతది — అంచనాలు తక్కువ ఖచ్చితంగా ఉండవచ్చు.",
        "low_data_warning":"⚠️ ఈ పంట/జిల్లాకు {n} డేటా పాయింట్లు మాత్రమే ఉన్నాయి. నమ్మకమైన అంచనాకు కనీసం 30 అవసరం.",
        "explanation_title": "ఈ ధర ఎందుకు?",
        "image_alt":        "{crop} పంట చిత్రం",
    }
}
