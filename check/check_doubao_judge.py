import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

api_key = os.getenv("DOUBAO_API_KEY")
base_url = os.getenv("DOUBAO_BASE_URL")
model = os.getenv("DOUBAO_MODEL")

print("DOUBAO_API_KEY exists:", bool(api_key))
print("DOUBAO_BASE_URL:", base_url)
print("DOUBAO_MODEL:", model)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

resp = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are a strict JSON generator. Return JSON only."
        },
        {
            "role": "user",
            "content": 'Return exactly this JSON: {"correct": true, "reason": "test passed"}'
        },
    ],
    temperature=0,
)

print(resp.choices[0].message.content)