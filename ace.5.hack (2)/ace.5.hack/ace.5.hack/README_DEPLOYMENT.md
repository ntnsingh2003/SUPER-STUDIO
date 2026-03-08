# Deploying AI Super Studio

This guide contains instructions on how to deploy this repository to production environments like Render, Railway, DigitalOcean, or any standard VPS.

## Preparing for Deployment

The application is structured as a **Monolith**: the FastAPI backend explicitly serves the `frontend` folder statically. This means the entire application (UI + API) runs on a single port (`8000` by default).

### 1. Environment Variables
You will need to set the following environment variables in your deployment platform's dashboard:
- `HUGGINGFACE_API_KEY`: Required. Get this from your HuggingFace account.

### 2. Docker Deployment (Recommended)
This repository includes a production-ready `Dockerfile`. Platforms like Render and Railway will automatically detect the Dockerfile and build the image for you.

Simply connect your GitHub repository to your host and select "Docker" as the build environment. The `Dockerfile` exposes port `8000`.

### 3. Native Python Deployment (Without Docker)
If you are deploying natively to a Linux VPS or a platform using standard Python Buildpacks (like Heroku):

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```
*(Make sure to swap `$PORT` with your environment's port variable, or `8000`)*
