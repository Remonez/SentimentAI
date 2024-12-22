# Use a Python base image
FROM python:3.12.4-slim

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY models ./models
COPY Gradio_app.py .
COPY requirements.txt .


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for the app
EXPOSE 7860

# Run the Gradio app
CMD ["python", "Gradio_app.py"]
