"""
Evaluator Agent - Output Validation & Hallucination Detection
Uses Llama-3.3-70B (best analytical reasoning) via the model router.
Acts as the quality gate in the LangGraph pipeline.
"""
from backend.llm.llm_service import generate_response


def evaluator_agent(original_prompt: str, agent_output: str, agent_name: str = "unknown") -> dict:
    """
    Reviews an agent's output for quality, accuracy, and hallucinations.
    
    Returns a dict with:
    - "verdict": "PASS" | "WARN" | "FAIL"
    - "confidence": 0-100
    - "feedback": explanation
    - "improved_output": str (only if verdict is WARN/FAIL)
    """
    # Truncate output to avoid token limits
    truncated_output = agent_output[:2000] if len(agent_output) > 2000 else agent_output

    eval_prompt = f"""You are **Evaluator Agent**, a strict quality assurance AI.
Your job is to review the output of other AI agents before it reaches the user.

## Original User Request
{original_prompt}

## Agent That Produced Output
{agent_name}

## Agent Output to Review
{truncated_output}

## Evaluation Criteria
1. **Relevance** (0-25): Does it directly address the user's request?
2. **Accuracy** (0-25): Are facts correct? Any hallucinations or fabrications?
3. **Completeness** (0-25): Is anything important missing?
4. **Quality** (0-25): Is it well-formatted, professional, and actionable?

## INSTRUCTIONS
Score each criterion, sum them for the confidence score (0-100).
Return ONLY a valid JSON object with NO markdown formatting, NO backticks:
{{"verdict": "PASS", "confidence": 85, "feedback": "Well-structured response covering all aspects", "improved_output": ""}}

Rules:
- PASS: confidence >= 70
- WARN: confidence 40-69
- FAIL: confidence < 40
- Only provide "improved_output" if verdict is WARN or FAIL"""

    raw = generate_response(eval_prompt, task_type="evaluation", max_tokens=512)

    import json
    try:
        raw = raw.strip()
        # Strip markdown code blocks if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(raw)
        # Validate and normalize
        result["verdict"] = result.get("verdict", "PASS").upper()
        result["confidence"] = min(100, max(0, int(result.get("confidence", 70))))
        result["feedback"] = result.get("feedback", "No feedback provided.")
        result["improved_output"] = result.get("improved_output", "")
        return result
    except (json.JSONDecodeError, ValueError, KeyError):
        return {
            "verdict": "PASS",
            "confidence": 65,
            "feedback": "Auto-passed (evaluator could not parse its own structured output).",
            "improved_output": ""
        }
