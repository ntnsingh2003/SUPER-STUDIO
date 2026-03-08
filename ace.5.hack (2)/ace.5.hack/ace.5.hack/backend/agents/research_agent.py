"""
Research Agent - Web Search & Summarization
Uses Llama-3.3-70B (best analytical reasoning) via the model router.
Includes a tool plugin system for extensible capabilities.
"""
from backend.llm.llm_service import generate_response

# ── Tool Plugin System ──────────────────────────────────────────────
# Tools are simple callables the agent can invoke.
# Architecture allows plugging in real APIs (Google, Bing, Serper, etc.)
TOOLS = {
    "web_search": {
        "name": "Web Search",
        "description": "Searches the web for information on a topic.",
        "execute": lambda query: f"[Simulated search results for: {query}]"
    },
    "summarize": {
        "name": "Text Summarizer",
        "description": "Summarizes long text into key points.",
        "execute": lambda text: generate_response(
            f"Summarize the following into 3-5 key bullet points:\n\n{text}",
            task_type="analysis", max_tokens=512
        )
    }
}


def research_agent(prompt: str, context: str = "") -> str:
    """Conducts deep research on a topic and returns a structured report."""
    search_results = TOOLS["web_search"]["execute"](prompt)

    system_prompt = f"""You are **Research Agent**, a world-class research analyst AI with expertise in:
- 🔍 Deep-dive research and literature review
- 📊 Data analysis and trend identification
- 🌐 Cross-domain knowledge synthesis
- 📝 Academic-quality report writing

## Available Tool Results
{search_results}

## Conversation Context
{context if context else "Fresh conversation."}

## Research Request
{prompt}

## Output Format (MUST follow this structure)
# [Research Topic Title]

## 📋 Executive Summary
A 2-3 sentence overview of your findings.

## 🔍 Key Findings
Detailed analysis organized by sub-topic, using:
- `###` headers for each finding area
- Bullet points for specific facts
- **Bold** for critical data points

## 📊 Analysis & Implications
What these findings mean and their broader impact.

## 🔮 Future Outlook
Predictions and emerging trends related to this topic.

## 📚 Key Takeaways
- Numbered list of the 3-5 most important conclusions

Format your ENTIRE response in rich Markdown."""

    return generate_response(system_prompt, task_type="analysis", max_tokens=1024)
