FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies required for PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY image_translator.py .

# Create a directory for input files
RUN mkdir -p /app/input

# Create volume mount points
VOLUME ["/app/input", "/app/output"]

# Copy Secret
COPY .env /app/.env

# Set environment variables if needed
ENV PYTHONUNBUFFERED=1

# Command to run the script
ENTRYPOINT ["python3", "image_translator.py"]

# Default arguments (can be overridden at runtime)
CMD ["--help"]