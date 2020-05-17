# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tiangolo/uvicorn-gunicorn-fastapi

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

EXPOSE 80

RUN python -m pip install --upgrade pip

# Install pip requirements
ADD serving-api/requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
ADD serving-api .
ADD util util