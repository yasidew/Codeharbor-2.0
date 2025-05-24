# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install system dependencies (for psycopg2 and PyMuPDF)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Run model download and Django app
CMD ["sh", "-c", "python download_model.py && python manage.py runserver 0.0.0.0:8000"]
