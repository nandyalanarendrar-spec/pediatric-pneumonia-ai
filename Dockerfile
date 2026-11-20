# Use official lightweight Python 3.11 image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and trained model
COPY app.py .
COPY resnet50v2_pro.keras .
COPY test_images/ ./test_images/

# Expose Gradio default port
EXPOSE 7860

# Ensure Gradio listens on all interfaces inside the container
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Run the clinical web app
CMD ["python", "app.py"]
