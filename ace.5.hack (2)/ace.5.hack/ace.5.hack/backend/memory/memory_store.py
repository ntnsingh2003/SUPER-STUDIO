"""
Memory System - Conversation & Context Store for AI Super Studio
Uses an in-memory store with optional ChromaDB persistence.
Provides per-session conversation history for multi-turn agent interactions.
"""
import time
import uuid
from collections import defaultdict
from typing import Optional

# ── In-Memory Conversation Store ────────────────────────────────────
# Uses a simple dict-based store. ChromaDB can be plugged in for
# persistent vector-based memory when the 'chromadb' package is available.
_conversation_store: dict[str, list[dict]] = defaultdict(list)
_metadata_store: dict[str, dict] = {}

# Optional ChromaDB integration
_chroma_client = None
_chroma_collection = None

try:
    import chromadb
    _chroma_client = chromadb.Client()
    _chroma_collection = _chroma_client.get_or_create_collection(
        name="agent_memory",
        metadata={"hnsw:space": "cosine"}
    )
    print("[Memory] ChromaDB initialized for persistent context.")
except ImportError:
    print("[Memory] ChromaDB not available. Using in-memory store only.")
except Exception as e:
    print(f"[Memory] ChromaDB init error: {e}. Using in-memory store.")


def create_session() -> str:
    """Create a new conversation session and return its ID."""
    session_id = str(uuid.uuid4())[:8]
    _metadata_store[session_id] = {
        "created_at": time.time(),
        "message_count": 0,
        "agents_used": []
    }
    return session_id


def add_message(session_id: str, role: str, content: str, agent_name: str = "system"):
    """Add a message to a conversation session."""
    if session_id not in _metadata_store:
        create_session_with_id(session_id)

    # Don't store massive base64 image strings in memory to avoid breaking token limits
    stored_content = content
    if isinstance(content, str) and content.startswith("data:image/"):
        stored_content = "[Image automatically removed from history to preserve context size]"

    message = {
        "role": role,
        "content": stored_content,
        "agent": agent_name,
        "timestamp": time.time()
    }
    _conversation_store[session_id].append(message)
    _metadata_store[session_id]["message_count"] += 1

    if agent_name not in _metadata_store[session_id]["agents_used"]:
        _metadata_store[session_id]["agents_used"].append(agent_name)

    # Also store in ChromaDB for semantic retrieval
    if _chroma_collection is not None:
        try:
            _chroma_collection.add(
                documents=[content],
                metadatas=[{"session_id": session_id, "role": role, "agent": agent_name}],
                ids=[f"{session_id}_{_metadata_store[session_id]['message_count']}"]
            )
        except Exception:
            pass  # Silently fail ChromaDB writes


def create_session_with_id(session_id: str):
    """Initialize a session with a specific ID."""
    _metadata_store[session_id] = {
        "created_at": time.time(),
        "message_count": 0,
        "agents_used": []
    }


def get_history(session_id: str, last_n: int = 10) -> list[dict]:
    """Retrieve the last N messages from a session."""
    return _conversation_store.get(session_id, [])[-last_n:]


def get_context_string(session_id: str, last_n: int = 6) -> str:
    """Format recent conversation history as a string for agent context."""
    history = get_history(session_id, last_n)
    if not history:
        return ""
    lines = []
    for msg in history:
        prefix = "User" if msg["role"] == "user" else f"Agent ({msg['agent']})"
        lines.append(f"{prefix}: {msg['content'][:300]}")
    return "\n".join(lines)


def search_memory(query: str, session_id: Optional[str] = None, n_results: int = 3) -> list[str]:
    """Search conversation memory using ChromaDB semantic similarity."""
    if _chroma_collection is None:
        return []
    try:
        where_filter = {"session_id": session_id} if session_id else None
        results = _chroma_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []


def get_all_sessions() -> list[dict]:
    """Return metadata for all active sessions."""
    sessions = []
    for sid, meta in _metadata_store.items():
        sessions.append({
            "session_id": sid,
            "message_count": meta["message_count"],
            "agents_used": meta["agents_used"],
            "created_at": meta["created_at"]
        })
    return sessions
