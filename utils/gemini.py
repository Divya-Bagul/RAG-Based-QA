import requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = "AIzaSyAzpbqWGBai_WmOVrKe_ZwR2e2yqA3tHTE"
# Configure genai with the API key
genai.configure(api_key=api_key)
def rewrite_text(raw_answer):
   
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Adding a friendly response starter
    full_prompt = f"""Rewrite the following answer in a simple and friendly way. Use only the rewritten text with no extra sentences or words. Add an example only if it helps clarify the meaning.

    Answer:
    {raw_answer}
    """

    response = model.generate_content(full_prompt)
    return response.text.strip()