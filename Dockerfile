# Use lightweight Python image
FROM python:3.11-slim

# Prevent Python from buffering output
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (important for ML + Django)
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Create media directory inside container
RUN mkdir -p /app/media

# Expose port
EXPOSE 8000

# Run Django (production-like)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
