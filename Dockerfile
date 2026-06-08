FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by Streamlit health checks and Python wheels.
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install only dashboard/runtime dependencies for online deployment.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-deploy.txt

# Copy app code and prepared artifacts. Heavy raw files are excluded by .dockerignore.
COPY . /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
