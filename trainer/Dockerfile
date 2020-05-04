# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow:nightly-gpu

### Tensorflow Debug logging
ENV TF_CPP_MIN_LOG_LEVEL 3
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN python -m pip install --upgrade pip

# Install pip requirements
ADD trainer/requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
ADD trainer .
ADD util util

USER root
# During debugging, this entry point will be overridden. For more information, refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "main.py"]