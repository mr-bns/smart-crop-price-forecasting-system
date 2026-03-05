"""
groq_ai.py  —  Groq LLM Advisory (optional, not used by main app)
==================================================================
This module is available for future use but is NOT currently called
by app.py — the built-in advisory engine in advisory.py is used instead.

To re-enable: import and call ai_explain() from app.py where needed.
"""

import os
import requests


def ai_explain(crop: str, price: float, lang: str = "en") -> str:
    """
    Generate a simple farmer advisory using Groq LLM.
    Returns a plain-text string, or a fallback message if unavailable.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return ""   # Silent fallback — caller should handle empty string

    lang_map = {"en": "English", "te": "Telugu"}
    language = lang_map.get(lang, "English")

    prompt = (
        f"Act as a friendly agricultural expert advising an Indian farmer.\n"
        f"Language: {language} — use simple, conversational words farmers easily understand.\n\n"
        f"Context:\n"
        f"  Crop: {crop}\n"
        f"  Predicted Price: ₹{price}/kg\n\n"
        f"Task:\n"
        f"1. Tell the farmer the predicted price clearly.\n"
        f"2. Give ONE piece of advice: sell now, wait, or monitor.\n\n"
        f"Keep it under 3 sentences. Do NOT cut off mid-sentence."
    )

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 150,
            },
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("choices"):
                return data["choices"][0]["message"]["content"].strip()
        return ""
    except Exception:
        return ""   # Silent fallback
