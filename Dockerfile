# 1. Start with Python 3.11.9(exact local version) on a slim Linux base
FROM python:3.11.9-slim

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy requirements first (Docker caching trick - when code change, no need to reinstall all dependencies)
COPY ./requirements.txt /code/requirements.txt

# 4. Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy your actual application code
COPY ./app /code/app

# 6. Command to run the server (Port 8080 for Google Cloud)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]