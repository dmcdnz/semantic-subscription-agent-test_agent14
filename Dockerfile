FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code and assets
COPY agent.py .
COPY agent_base.py .
COPY agent_container.py .
COPY interest_model.py .
COPY config.yaml .

# Create an empty examples file that will be overwritten if one exists
RUN touch /app/examples.jsonl

# Try to copy examples file (will override the empty one)
COPY examples.jsonl /app/

# Create directory for models and copy any existing fine-tuned model files
RUN mkdir -p ./fine_tuned_model/

# Create necessary directories for the fine-tuned model
RUN mkdir -p ./fine_tuned_model/1_Pooling/

# Option 1: Try to copy all model files at once (might lead to Docker build warnings if files don't exist)
# Create directory for fine-tuned model (will be populated during build)
RUN mkdir -p ./fine_tuned_model/

# Add a README in the fine_tuned_model directory with instructions
RUN echo "# Fine-tuned Model Directory\n\nPlace your fine-tuned sentence transformer model files here to improve agent interest detection.\nThe agent will automatically use this model when present." > ./fine_tuned_model/README.md || echo "Copying fine_tuned_model directory"

# Option 2: Copy critical files individually (more reliable)
# Note: We ignore errors with || true to avoid build failures
COPY fine_tuned_model/classification_head.pt ./fine_tuned_model/ || true
COPY fine_tuned_model/model.safetensors ./fine_tuned_model/ || true
COPY fine_tuned_model/*.json ./fine_tuned_model/ || true
COPY fine_tuned_model/*.txt ./fine_tuned_model/ || true

# Make sure we have a proper README for debugging
RUN echo "This directory should contain classification_head.pt and other model files" > ./fine_tuned_model/README.md

# Copy pooling directory files if they exist
COPY fine_tuned_model/1_Pooling/* ./fine_tuned_model/1_Pooling/ || true

# Environment variables
ENV AGENT_ID="${AGENT_ID}"
ENV AGENT_NAME="${AGENT_NAME}"

# Set Core API URL - use host.docker.internal for Windows/Mac, host network for Linux
# For Linux compatibility, this will be overridden at runtime
ENV CORE_API_URL="http://host.docker.internal:8888"

# Run the agent
CMD ["python", "agent_container.py"]
