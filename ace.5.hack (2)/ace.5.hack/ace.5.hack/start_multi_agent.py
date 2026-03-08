import subprocess
import os
import sys
import time
import http.server
import socketserver
import threading

base_dir = os.path.dirname(os.path.abspath(__file__))
env = os.environ.copy()
env["PYTHONPATH"] = base_dir

print("🚀 Starting AI Super Studio – Multi-Agent Platform...")

# 1. Start Unified Backend
print("Starting Unified Backend on port 8000...")
backend_cmd = [sys.executable, "-m", "backend.main"]
backend_proc = subprocess.Popen(backend_cmd, cwd=base_dir, env=env)

# 2. Start Frontend Server (using simple http server)
PORT = 3000
FRONTEND_DIR = os.path.join(base_dir, "frontend")

def serve_frontend():
    os.chdir(FRONTEND_DIR)
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving Dashboard at http://127.0.0.1:{PORT}")
        httpd.serve_forever()

frontend_thread = threading.Thread(target=serve_frontend, daemon=True)
frontend_thread.start()

print("\nAll systems active!")
print(f"Dashboard: http://127.0.0.1:{PORT}/index.html")
print("API Docs: http://127.0.0.1:8000/docs")
print("\nPress Ctrl+C to terminate.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down...")
    backend_proc.terminate()
    print("Goodbye!")
