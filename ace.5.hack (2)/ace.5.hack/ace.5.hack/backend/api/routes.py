"""
API Routes - Unified Entry Points for AI Super Studio
Provides REST endpoints for workflow execution, streaming, memory, and health.
"""
import json
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from backend.orchestrator.langgraph_router import execute_workflow
from backend.orchestrator.agent_router import route_to_agent
from backend.llm.llm_service import stream_response
from backend.rag.vector_store import add_documents
from backend.memory.memory_store import (
    create_session, get_history, get_all_sessions, add_message
)

router = APIRouter()


# ── Request/Response Models ─────────────────────────────────────────

class WorkflowRequest(BaseModel):
    service: Optional[str] = None
    prompt: str
    session_id: Optional[str] = None
    use_planner: bool = False       # Set True to enable LangGraph orchestration
    optimize_prompt: bool = False   # Set True to optimize the prompt first

class IngestRequest(BaseModel):
    documents: list[str]

class StreamRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    service: Optional[str] = None


# ── Core Workflow Endpoints ─────────────────────────────────────────

@router.post("/execute-workflow")
async def api_execute_workflow(payload: WorkflowRequest):
    """
    Unified entry point for all Multi-Agent AI workflows.
    Supports both direct routing and LangGraph-style orchestration.
    """
    try:
        session_id = payload.session_id or create_session()

        # Run the potentially blocking orchestrator in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: execute_workflow(
                prompt=payload.prompt,
                session_id=session_id,
                service=payload.service,
                use_planner=payload.use_planner,
                optimize_prompt=payload.optimize_prompt
            )
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream-workflow")
async def api_stream_workflow(payload: StreamRequest):
    """
    Streaming endpoint: Returns an SSE stream of LLM tokens.
    The frontend can consume this for real-time typing effects.
    """
    session_id = payload.session_id or create_session()
    add_message(session_id, "user", payload.prompt)

    # Build the prompt with optional agent context
    agent_prefix = ""
    if payload.service:
        agent_prefix = f"You are acting as the {payload.service} agent. "

    full_prompt = agent_prefix + payload.prompt

    async def event_generator():
        collected = []
        async for chunk in stream_response(full_prompt):
            collected.append(chunk)
            # Format as SSE event
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        
        # Store full response in memory
        full_response = "".join(collected)
        add_message(session_id, "assistant", full_response, agent_name=payload.service or "stream")
        yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ── Memory & Session Endpoints ──────────────────────────────────────

@router.post("/session/create")
async def api_create_session():
    """Create a new conversation session."""
    session_id = create_session()
    return {"session_id": session_id}

@router.get("/session/{session_id}/history")
async def api_get_history(session_id: str, last_n: int = 20):
    """Get conversation history for a session."""
    history = get_history(session_id, last_n)
    return {"session_id": session_id, "messages": history}

@router.get("/sessions")
async def api_list_sessions():
    """List all active conversation sessions."""
    return {"sessions": get_all_sessions()}


# ── RAG Endpoints ───────────────────────────────────────────────────

@router.post("/rag/ingest")
async def ingest_documents(payload: IngestRequest):
    """Ingest new facts into the FAISS vector store."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, add_documents, payload.documents)
        return {"status": "success", "message": f"Ingested {len(payload.documents)} documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health & Status ─────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """System health check with component status."""
    from backend.llm.llm_service import client as llm_client
    return {
        "status": "ok",
        "platform": "AI Super Studio v2.0",
        "components": {
            "llm_engine": "online" if llm_client else "offline",
            "memory": "online",
            "orchestrator": "online",
        }
    }
