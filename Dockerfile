FROM python:3.11-slim

# Install system dependencies as root (no sudo!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user
ENV HOME=/home/user
WORKDIR $HOME/app

# Set dynamic mode environment variable (default to cloud mode)
ARG IS_LOCAL=false
ENV IS_LOCAL=${IS_LOCAL}

# Install Python dependencies first (better layer caching)
COPY --chown=user requirements.txt .
# Install local mode dependencies if IS_LOCAL is true
COPY --chown=user requirements-dev.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && \
    if [ "$IS_LOCAL" = "true" ]; then \
        pip install --no-cache-dir -r requirements-dev.txt; \
    fi

# Copy the application
COPY --chown=user . .

# Download Vietnamese translation model
RUN python vi/download.py

# Hugging Face cache setup
ENV HF_HOME="$HOME/.cache/huggingface"
ENV SENTENCE_TRANSFORMERS_HOME="$HOME/.cache/huggingface/sentence-transformers"
ENV MEDGEMMA_HOME="$HOME/.cache/huggingface/sentence-transformers"

# Prepare runtime dirs
RUN mkdir -p $HOME/app/logs $HOME/app/cache $HOME/app/cache/hf $HOME/app/cache/outputs $HOME/app/data && \
    chown -R user:user $HOME/app

# Download MedAlpaca model if in local mode
RUN if [ "$IS_LOCAL" = "true" ]; then \
        echo "Downloading MedAlpaca-13b model for local mode..."; \
        python -c "from huggingface_hub import snapshot_download; import os; snapshot_download('medalpaca/medalpaca-13b', token=os.getenv('HF_TOKEN'), cache_dir='$HOME/.cache/huggingface')"; \
    fi

USER user

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
