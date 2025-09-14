FROM python:3.11-slim

# Install system dependencies as root (no sudo!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user
ENV HOME=/home/user
WORKDIR $HOME/app

# Install Python dependencies first (better layer caching)
COPY --chown=user requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY --chown=user . .

# Hugging Face cache setup
ENV HF_HOME="$HOME/.cache/huggingface"
ENV SENTENCE_TRANSFORMERS_HOME="$HOME/.cache/huggingface/sentence-transformers"
ENV MEDGEMMA_HOME="$HOME/.cache/huggingface/sentence-transformers"

# Prepare runtime dirs
RUN mkdir -p $HOME/app/logs $HOME/app/cache $HOME/app/cache/hf $HOME/app/cache/outputs && \
    chown -R user:user $HOME/app

USER user

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
