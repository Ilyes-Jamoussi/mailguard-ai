# Minimal CPU image for serving the MailGuard AI Streamlit demo.
# Same pattern as minigpt-llm: CPU-only torch wheel first, non-root user, $PORT honored.
FROM python:3.13-slim

WORKDIR /app

# Install the CPU-only torch wheel first (much smaller than the default CUDA
# build), then the rest of the pinned runtime stack.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Application code, UI assets, and trained artifacts
# (src/config.py resolves models/ and assets/ relative to the project root).
COPY app.py .
COPY src ./src
COPY models ./models
COPY assets ./assets

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV PYTHONPATH=/app \
    PORT=8501

EXPOSE 8501

# Honor the platform-provided $PORT and default to 8501 (Streamlit's default).
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
