# Use python 3.12 to ensure audioop compatibility
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g. for audio processing)
RUN apt-get update && apt-get install -y gcc

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Render sets the PORT env var automatically.
# We pass it to the server script implicitly via os.environ in python.
CMD ["python", "server.py"]