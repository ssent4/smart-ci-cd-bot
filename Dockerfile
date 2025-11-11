# Use official Python base
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy project
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt pytest

# Expose FastAPI port
EXPOSE 8000

# Default command (can override in docker run)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
