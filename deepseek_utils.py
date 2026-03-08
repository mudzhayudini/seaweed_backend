import os
import time
import requests


DEEPSEEK_API_KEY = os.getenv("sk-6f87b76fc46142d8bdc81347a0f55525")


def call_deepseek_api(
    prompt: str,
    system_message: str,
    temperature: float = 0.2,
    max_tokens: int = 350,
    retries: int = 3,
) -> str:
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API key is not configured on the server."

    url = "https://api.deepseek.com/chat/completions"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    last_error = None

    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"]

        except Exception as e:
            last_error = e
            print(f"DeepSeek attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    return f"DeepSeek API failed after multiple retries. Last error: {last_error}"