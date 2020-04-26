path=/home/$(whoami)/container-data/logs/
mkdir -p $path
docker build -t trainer:latest .
docker run -it --rm --gpus all -v $path:/logs trainer:latest