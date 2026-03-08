import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Use a lightweight embedding model
embeddings = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    print(f"CRITICAL Warning: Failed to load HuggingFaceEmbeddings: {e}")

# Sample Knowledge Base for the AI Studio
KNOWLEDGE_BASE = [
    Document(page_content="Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969.", metadata={"source": "history"}),
    Document(page_content="The sky appears blue because of Rayleigh scattering, where shorter wavelengths (blue) scatter more easily.", metadata={"source": "science"}),
    Document(page_content="Generative AI microservices can be orchestrated using LangChain agents for efficient task routing.", metadata={"source": "tech"}),
    Document(page_content="Python FastAPI is high-performance, easy to learn, fast to code, and ready for production.", metadata={"source": "docs"}),
]

vector_db = None

def get_vector_db():
    global vector_db
    if vector_db is None and embeddings is not None:
        try:
            vector_db = FAISS.from_documents(KNOWLEDGE_BASE, embeddings)
            print("FAISS Vector DB initialized.")
        except Exception as e:
            print(f"Error initializing FAISS: {e}")
    return vector_db

def retrieve_context(query: str, k: int = 2) -> str:
    db = get_vector_db()
    if not db:
        return ""
    
    try:
        docs = db.similarity_search(query, k=k)
        return "\n".join([d.page_content for d in docs])
    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""

def add_documents(texts: list[str]):
    global vector_db
    db = get_vector_db()
    if not db:
        print("Error: Cannot add documents, Vector DB is not initialized.")
        return
    
    new_docs = [Document(page_content=t, metadata={"source": "user_ingest"}) for t in texts]
    db.add_documents(new_docs)
    print(f"Successfully ingested {len(texts)} new documents into RAG.")
