# Dockerfile for a Python application using FastAPI and Uvicorn

# Start from Ubuntu:22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first to leverage Docker caching
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all necessary files (note the trailing slashes for directory destinations)
COPY . .

# Try to copy q_table.pkl if it exists
COPY q_table.pkl /app/

RUN chmod +x main.py

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the Python application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]