"""
Image Generation Agent - Vision Genesis
Uses HuggingFace text-to-image models to generate images from prompts.
Returns base64-encoded PNG images for frontend rendering.
"""
import os
import io
import base64
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(token=hf_token)

# ── Image Model Options (ordered by speed/reliability) ──────────────
IMAGE_MODELS = [
    "black-forest-labs/FLUX.1-schnell",                  # Fast & high quality
    "stabilityai/stable-diffusion-xl-base-1.0",          # SDXL fallback
]


def image_gen_agent(prompt: str) -> str:
    """
    Generates an image from a text prompt.
    Returns a base64-encoded data URI string (data:image/png;base64,...).
    Tries multiple models for reliability.
    """
    if not hf_token:
        return "Error: HUGGINGFACEHUB_API_TOKEN is not set in environment."

    # Enhance the prompt for better image generation
    enhanced_prompt = f"high quality, detailed, professional photograph, {prompt}"

    for model_id in IMAGE_MODELS:
        try:
            print(f"[ImageGen] Trying model: {model_id}")
            image = client.text_to_image(
                enhanced_prompt,
                model=model_id,
            )

            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            print(f"[ImageGen] Success with {model_id}")
            return f"data:image/png;base64,{img_str}"

        except Exception as e:
            error_msg = str(e)
            print(f"[ImageGen] {model_id} failed: {error_msg[:100]}")
            continue

    # All models failed
    return "Error: Image generation failed with all available models. The models may be loading — please try again in a moment."
