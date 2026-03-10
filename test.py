import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://openrouter.ai",
    "X-Title": "OpenRouter",
  },
  data=json.dumps({
    "model": "openai/gpt-5.2-chat",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
        ]
      }
    ]
  })
)
print(response.json())