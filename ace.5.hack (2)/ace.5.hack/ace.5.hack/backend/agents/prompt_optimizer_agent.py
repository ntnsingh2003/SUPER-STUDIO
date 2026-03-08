"""
Prompt Optimizer Agent - Intelligent Prompt Engineering
Uses Mistral-7B (fast and good at instruction-following) via the model router.
"""
from backend.llm.llm_service import generate_response


def prompt_optimizer_agent(prompt: str, target_agent: str = "general") -> str:
    """
    Takes a user's raw prompt and returns an optimized version
    tailored for the target agent.
    """
    system_prompt = f"""You are **Prompt Optimizer**, an AI specialist in prompt engineering.
Your ONLY job is to take a user's raw prompt and rewrite it to get better results.

## Target Agent
The optimized prompt will be sent to: **{target_agent}**

## Original User Prompt
{prompt}

## Optimization Strategy
1. Make the prompt more specific and detailed
2. Add relevant constraints and format requirements
3. Include context clues that help the target agent perform better
4. Preserve the user's original intent exactly
5. Add structure (e.g., "Step 1:", "Include:", "Format as:")

## CRITICAL RULES
- Return ONLY the improved prompt text
- Do NOT add explanations, meta-commentary, or "Here's the improved version:"
- Do NOT wrap in quotes or markdown
- Output the refined prompt directly, as if the user typed it themselves"""

    optimized = generate_response(system_prompt, task_type="planning", max_tokens=400)

    # Clean up: remove any meta-commentary the model might add
    optimized = optimized.strip()
    # Remove common prefixes the model might add
    for prefix in ["Here's", "Here is", "Optimized:", "Improved:", "Refined:"]:
        if optimized.startswith(prefix):
            optimized = optimized[len(prefix):].lstrip(": \n")

    return optimized if len(optimized) > 10 else prompt  # Fallback to original
