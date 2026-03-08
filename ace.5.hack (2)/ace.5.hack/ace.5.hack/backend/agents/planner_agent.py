"""
Planner Agent - Workflow Orchestration Brain
Uses Qwen2.5-72B for structured JSON planning.
Decides which agents to invoke (up to 3) and in what order.
"""
from backend.llm.llm_service import generate_response

PLANNER_SYSTEM_PROMPT = """You are the **Planner Agent** for AI Super Studio, a multi-agent AI platform.
Your ONLY job is to analyze the user's request and decide which agent(s) should handle it.

## Available Agents
| Agent ID | Specialty |
|----------|-----------|
| creative | Art, design, creative writing, poetry, narratives |
| codebase | Code generation, debugging, architecture, algorithms |
| marketing | Ad copy, campaigns, product descriptions, sales strategy |
| hallucination-auditor | Fact-checking, claim verification, truth analysis |
| synthetic-data | JSON dataset generation, mock data |
| image-gen | Image generation from text descriptions |
| research | Topic research, analysis, summarization, trend reports |

## Pipeline Strategy
You MUST chain 2-3 agents when the request benefits from multiple perspectives:

### Common Multi-Agent Pipelines
- "Research X and write about it" → ["research", "creative"] (research first, then creative writing)
- "Research X and create a marketing pitch" → ["research", "marketing"] (research, then pitch)
- "Research, write, and sell" → ["research", "creative", "marketing"] (full 3-agent pipeline)
- "Write code and explain it" → ["codebase", "creative"] (code, then explanation)
- "Analyze data trends and create a report" → ["research", "codebase", "creative"]
- "Fact-check and summarize" → ["hallucination-auditor", "research"]
- "Generate data and analyze it" → ["synthetic-data", "codebase"]

### When to Use 1 Agent
- Simple, focused tasks: "write a poem" → ["creative"]
- Direct code requests: "write a sort function" → ["codebase"]
- Image requests: "draw a cat" → ["image-gen"]
- Pure fact-check: "is X true?" → ["hallucination-auditor"]

### When to Use 2-3 Agents
- The request mentions multiple tasks (research AND write, analyze AND present)
- The request is complex and benefits from research before creation
- The request involves creating professional content (research → write → polish)

## Rules
- Chain agents in LOGICAL ORDER (research before writing, data before analysis)
- Maximum 3 agents in a pipeline
- set should_evaluate to true for factual or critical tasks
- ALWAYS return valid JSON

## Response Format
Return ONLY a valid JSON object. NO markdown. NO backticks. NO explanation.
{"agents": ["agent-1", "agent-2", "agent-3"], "reasoning": "why this pipeline", "should_evaluate": true}
"""


def planner_agent(prompt: str, context: str = "") -> dict:
    """Analyzes the user prompt and returns a multi-agent workflow plan."""
    full_prompt = f"""{PLANNER_SYSTEM_PROMPT}

Conversation context: {context if context else "None"}

User request: {prompt}

Return ONLY the JSON object:"""

    raw = generate_response(full_prompt, task_type="planning", max_tokens=250)

    import json
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        plan = json.loads(raw)
        if not isinstance(plan.get("agents"), list) or len(plan["agents"]) == 0:
            raise ValueError("No agents")
        # Enforce max 3, deduplicate
        seen = []
        for a in plan["agents"]:
            if a not in seen and len(seen) < 3:
                seen.append(a)
        plan["agents"] = seen
        plan.setdefault("reasoning", "Planner routed request")
        plan.setdefault("should_evaluate", False)
        return plan
    except (json.JSONDecodeError, ValueError):
        return _fallback_routing(prompt)


def _fallback_routing(prompt: str) -> dict:
    """Keyword-based fallback with multi-agent pipeline support."""
    p = prompt.lower()

    # Multi-agent patterns (check first)
    if any(kw in p for kw in ["research", "find out", "look up"]) and any(kw in p for kw in ["write", "create", "compose", "draft"]) and any(kw in p for kw in ["market", "pitch", "sell", "advertise", "promote"]):
        return {"agents": ["research", "creative", "marketing"], "reasoning": "Research + Write + Market pipeline", "should_evaluate": True}

    if any(kw in p for kw in ["research", "find out", "look up", "learn about"]) and any(kw in p for kw in ["write", "create", "compose", "draft", "poem", "story", "essay"]):
        return {"agents": ["research", "creative"], "reasoning": "Research then creative writing pipeline", "should_evaluate": False}

    if any(kw in p for kw in ["research", "find out", "look up"]) and any(kw in p for kw in ["market", "pitch", "sell", "ad", "campaign", "promote"]):
        return {"agents": ["research", "marketing"], "reasoning": "Research then marketing pipeline", "should_evaluate": False}

    if any(kw in p for kw in ["code", "program", "build", "implement"]) and any(kw in p for kw in ["explain", "describe", "document", "write about"]):
        return {"agents": ["codebase", "creative"], "reasoning": "Code then explain pipeline", "should_evaluate": False}

    if any(kw in p for kw in ["data", "generate data", "dataset"]) and any(kw in p for kw in ["analyze", "analysis", "code", "visualize"]):
        return {"agents": ["synthetic-data", "codebase"], "reasoning": "Data then analyze pipeline", "should_evaluate": False}

    # Single agent patterns
    if any(kw in p for kw in ["image", "picture", "draw", "generate art", "illustration", "photo", "visual"]):
        return {"agents": ["image-gen"], "reasoning": "Image generation", "should_evaluate": False}
    elif any(kw in p for kw in ["code", "function", "debug", "python", "javascript", "program", "algorithm", "api", "implement"]):
        return {"agents": ["codebase"], "reasoning": "Code request", "should_evaluate": False}
    elif any(kw in p for kw in ["fact", "true", "false", "verify", "check", "claim", "accurate"]):
        return {"agents": ["hallucination-auditor"], "reasoning": "Fact-checking", "should_evaluate": True}
    elif any(kw in p for kw in ["market", "ad", "campaign", "sell", "product", "copy", "brand", "slogan", "pitch"]):
        return {"agents": ["marketing"], "reasoning": "Marketing", "should_evaluate": False}
    elif any(kw in p for kw in ["data", "json", "dataset", "synthetic", "fake data", "mock"]):
        return {"agents": ["synthetic-data"], "reasoning": "Data generation", "should_evaluate": False}
    elif any(kw in p for kw in ["research", "search", "find", "summarize", "article", "study", "analysis"]):
        return {"agents": ["research"], "reasoning": "Research", "should_evaluate": True}
    else:
        return {"agents": ["creative"], "reasoning": "General task", "should_evaluate": False}
