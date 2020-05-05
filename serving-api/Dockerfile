# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow:nightly

ENV TF_CPP_MIN_LOG_LEVEL 2

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

EXPOSE 5000

RUN python -m pip install --upgrade pip

# Install pip requirements
ADD serving-api/requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
ADD serving-api .
ADD util util

# During debugging, this entry point will be overridden. For more information, refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "main.py"]