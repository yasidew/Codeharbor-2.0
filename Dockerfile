# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install system dependencies (e.g., for psycopg2 and PyMuPDF)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Collect static files (optional for Django)
#RUN python manage.py collectstatic --noinput

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Run the app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]



## Use Python base image
#FROM python:3.12-slim
#
## Set environment variables
#ENV PYTHONDONTWRITEBYTECODE 1
#ENV PYTHONUNBUFFERED 1
#
## Set work directory
#WORKDIR /app
#
## Install dependencies
#COPY requirements.txt .
#RUN pip install --upgrade pip
#RUN pip install -r requirements.txt
#
## Copy project
#COPY . .
#
## Default command (will be overridden in docker-compose)
#CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
