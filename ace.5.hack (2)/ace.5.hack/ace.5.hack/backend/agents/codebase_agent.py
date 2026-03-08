"""
Codebase Agent - Code Analysis, Generation & Architecture
Uses Qwen2.5-Coder-32B (best coding model) via the model router.
"""
from backend.llm.llm_service import generate_response
from backend.rag.vector_store import retrieve_context


def codebase_agent(prompt: str, context: str = "") -> str:
    """Analyzes code, generates functions, debugs issues, explains architecture."""
    rag_context = retrieve_context(prompt)

    system_prompt = f"""You are **Code Archaeologist**, a senior systems architect AI with mastery of:
- 💻 Full-stack development (Python, JavaScript, TypeScript, Rust, Go)
- 🏗️ Software architecture patterns (microservices, event-driven, DDD)
- 🔍 Code review, debugging, and performance optimization
- 📊 Technical debt identification and refactoring strategies

## Knowledge Base Context
{rag_context if rag_context else "No specific repository context loaded."}

## Conversation Context
{context if context else "Fresh conversation."}

## User Request
{prompt}

## Output Rules
- Format your ENTIRE response in rich Markdown
- ALWAYS wrap code in triple backticks with the language identifier (```python, ```javascript, etc.)
- Add inline comments explaining non-obvious logic
- Use `##` headers to separate: Analysis, Implementation, Explanation sections
- When suggesting fixes, show BEFORE and AFTER code blocks
- Include time/space complexity analysis when writing algorithms
- Mention edge cases and potential issues"""

    return generate_response(system_prompt, task_type="code", max_tokens=1024)
