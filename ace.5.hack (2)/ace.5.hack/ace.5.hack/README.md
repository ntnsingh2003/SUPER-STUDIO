# ⚡ AI Super Studio

**AI Super Studio** is a production-grade, multi-agent AI platform featuring LangGraph-style workflow orchestration, conversation memory, and a dynamic frontend UI. It utilizes a network of specialized HuggingFace LLMs to collaboratively solve complex tasks like planning, generating creative content, synthesizing codebase architecture, and producing image prompts.

## 🌟 Features
- **Auto-Planner:** Automatically parses human prompts and dynamically chains up to 3 specialized agents to handle the objective.
- **Workflow Visualizer:** See exactly which agents are acting, their generated status steps, and pipeline context in real-time.
- **Agent Collaboration Context:** Each agent in the chain receives the context produced by the previous agent in the pipeline, ensuring smooth, integrated handoffs.
- **Multi-Modal Generation:** Supports returning UI-rendered assets like Base64 images directly inside workflows.
- **Fully Asynchronous UI:** Streaming responses with SSE and typing indicators powered by React.

## 🧠 Pre-configured Agents
- **`planner`**: Analyzes the request and determines the optimal pipeline flow.
- **`image-gen`**: Generates image artifacts and returns Base64 representations.
- **`creative`**: Handles copywriting, poetry, and storytelling.
- **`codebase`**: Specialized software engineering and architecture node.
- **`marketing`**: Extracts demographics and constructs campaign copy.
- **`hallucination-auditor`**: Fact checks output for strict veracity.
- **`research`**: Compiles data, statistics, and references.
- **`synthetic-data`**: Outputs JSON-structured synthetic datasets.
- **`prompt-optimizer`**: Evaluates and improves the human's initial system prompt.

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, Uvicorn, LangChain, HuggingFace InferenceClient
- **Frontend:** React, TailwindCSS, PrismJS, Marked
- **Memory Store:** In-memory context array with ChromaDB persistent fallback.

## 🚀 Quick Start (Local Setup)

### 1. Requirements
Ensure you have **Python 3.11+** installed.

### 2. Environment Configuration
Clone the repository and create an `.env` file from the sample:
```bash
git clone https://github.com/ntnsingh2003/SUPER-STUDIO.git
cd SUPER-STUDIO
cp .env.example .env
```
Add your free [HuggingFace API Key](https://huggingface.co/settings/tokens) to the `.env` file:
```ini
HUGGINGFACE_API_KEY=hf_your_token_here
```

### 3. Installation & Run
Install the dependencies:
```bash
pip install -r requirements.txt
```
Start the application (Full Stack via one FastAPI port):
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Then simply open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

## 🐳 Deployment (Docker)
The `.dockerignore` and `Dockerfile` are completely configured. Deploy it instantly to Render, Railway, or VPS by linking standard Docker settings and injecting the `HUGGINGFACE_API_KEY` environment variable. See `README_DEPLOYMENT.md` for more configuration information.

## 📄 License
This project is open source and available for modification.
