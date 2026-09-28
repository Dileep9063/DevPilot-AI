from django.conf import settings
from google import genai


client = genai.Client(api_key=settings.GEMINI_API_KEY)


def ask_gemini(message):
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=message,
    )

    return response.text