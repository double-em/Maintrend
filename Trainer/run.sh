path=/home/$(whoami)/container-data/logs/$(date '+%F@%H;%M;%S')
mkdir -p $path
docker build -t trainer:latest .
docker run -it --rm --gpus all -v $path:/logs trainer:latest