"""
LLM Service - Core AI Engine for AI Super Studio
Provides synchronous, asynchronous, and streaming LLM interactions
via the HuggingFace Inference API.
"""
import os
import asyncio
from functools import lru_cache
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ── Model Registry ──────────────────────────────────────────────────
# Maps task categories to the best-suited HuggingFace Inference API model.
# All models below have been VERIFIED to work with this API token.
MODEL_REGISTRY = {
    "general":    "meta-llama/Llama-3.3-70B-Instruct",       # Best overall reasoning (70B)
    "code":       "Qwen/Qwen2.5-Coder-32B-Instruct",         # Best for code generation
    "creative":   "Qwen/Qwen2.5-72B-Instruct",               # Strong creative writing (72B)
    "analysis":   "meta-llama/Llama-3.3-70B-Instruct",       # Deep analytical reasoning
    "planning":   "Qwen/Qwen2.5-72B-Instruct",               # Excellent structured JSON output
    "evaluation": "meta-llama/Llama-3.3-70B-Instruct",       # Careful factual evaluation
}

# ── Client Initialization ───────────────────────────────────────────
try:
    client = InferenceClient(token=hf_token)
    print("[LLM] HuggingFace InferenceClient initialized.")
except Exception as e:
    print(f"[LLM] Initialization Error: {e}")
    client = None


def select_model(task_type: str = "general") -> str:
    """Model Router: Selects the best model based on the task type."""
    return MODEL_REGISTRY.get(task_type, MODEL_REGISTRY["general"])


def generate_response(prompt: str, task_type: str = "general", max_tokens: int = 1024) -> str:
    """Synchronous LLM call (blocking). Used by legacy agent functions."""
    if not client:
        return "LLM Backend Error: Model failed to initialize."
    try:
        model = select_model(task_type)
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        import traceback
        return f"LLM Generation Error: {repr(e)}\n\n{traceback.format_exc()}"


async def agenerate_response(prompt: str, task_type: str = "general", max_tokens: int = 1024) -> str:
    """Asynchronous LLM call. Runs the blocking HF call in a thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_response, prompt, task_type, max_tokens)


async def stream_response(prompt: str, task_type: str = "general", max_tokens: int = 1024):
    """
    Async generator that yields text chunks from the LLM.
    Uses the HuggingFace streaming API for real-time token delivery.
    """
    if not client:
        yield "LLM Backend Error: Model failed to initialize."
        return
    try:
        model = select_model(task_type)
        messages = [{"role": "user", "content": prompt}]
        stream = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"\n\n[Streaming Error]: {repr(e)}"