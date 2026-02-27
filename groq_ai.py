import requests
import os

# Get API Key securely
API_KEY = os.environ.get("GROQ_API_KEY")

def ai_explain(crop, price, lang="en"):
    """
    Generate a simple explanation for the farmer using Groq AI.
    """
    if not API_KEY:
        return "⚠ AI Key Missing. Please check .env file."

    # Map language code to name
    lang_map = {
        "en": "English",
        "hi": "Hindi",
        "te": "Telugu"
    }
    language = lang_map.get(lang, "English")

    prompt = f"""
    Act as a friendly agricultural expert advising a farmer.
    Language: {language} (Use simple, daily spoken words that farmers easily understand).

    Context:
    - Crop: {crop}
    - Predicted Price: ₹{price}/kg

    Task:
    1. Tell the farmer the price clearly.
    2. Give 1 piece of advice (sell now or wait).
    
    Constraints:
    - Keep it short (maximum 2-3 sentences).
    - Do NOT use complex bookish translation. Use natural conversational tone.
    - CRITICAL: Ensure the response is COMPLETE. Do not cut off in the middle. Finish your sentence.
    """

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500 # Increased to prevent unfinished sentences
            },
            timeout=10
        )

        if response.status_code != 200:
            return f"⚠ AI unavailable (Status {response.status_code})"

        data = response.json()
        
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
            
        return "⚠ AI response empty."

    except Exception:
        return "⚠ AI connection failed."
