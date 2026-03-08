import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv('.env')
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

models = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]

client = InferenceClient(token=token)

for m in models:
    print(f"Testing {m}...")
    try:
        res = client.chat_completion(messages=[{"role": "user", "content": "hi"}], model=m, max_tokens=10)
        print(f"SUCCESS with {m}: {res.choices[0].message.content}")
        break
    except Exception as e:
        print(f"FAILED {m}: {e}")
