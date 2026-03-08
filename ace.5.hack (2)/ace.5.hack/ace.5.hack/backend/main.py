"""
AI Super Studio v2.0 - Production-Grade Multi-Agent Platform
Main application entry point with FastAPI server configuration.
"""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.api.routes import router

app = FastAPI(
    title="AI Super Studio v2.0 - Multi-Agent Platform",
    description="Production-grade agentic AI platform with LangGraph-style orchestration, "
                "streaming responses, conversation memory, and multi-agent collaboration.",
    version="2.0.0"
)

# ── CORS Middleware ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Router ──────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")

# ── Static Frontend ─────────────────────────────────────────────────
# Serve the frontend directory so the app works seamlessly on a single port for production
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"[Warning] Frontend directory not found at {frontend_dir}. UI will not be served.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
