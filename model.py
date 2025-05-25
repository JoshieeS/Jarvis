# openai_client.py

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# 1. Read your GitHub token and endpoint from the environment
API_KEY  = os.getenv("OPENAI_API_KEY")
ENDPOINT = "https://models.github.ai/inference"
MODEL_ID = "openai/gpt-4.1"

# 2. Instantiate the OpenAI client with your custom base URL
client = OpenAI(api_key=API_KEY, base_url=ENDPOINT)

def ask_openai(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 1.0,
    top_p: float = 1.0
) -> str:
    """
    Send a chat-completion request, allowing a custom system prompt.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    return resp.choices[0].message.content.strip()