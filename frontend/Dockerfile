# 1. Use a lightweight Python base
FROM python:3.11.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Install just the libraries we need for the UI
RUN pip install --no-cache-dir streamlit requests

# 4. Copy the app code into the container
COPY app.py .

# 5. Run Streamlit
# CRITICAL: We must tell Streamlit to run on Port 8080 because Google Cloud expects it.
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]