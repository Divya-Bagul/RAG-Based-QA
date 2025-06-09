import requests
import os

GROQ_API_KEY = "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"  # replace with env or secret manager
MODEL = "llama-3.3-70b-versatile"#"meta-llama/llama-4-scout-17b-16e-instruct" #"llama3-70b-8192"  # or mixtral-8x7b

def fallback_response(question):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.7
    }
    try:
        res = requests.post(url, json=payload, headers=headers)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Groq LLM failed: {str(e)}"
