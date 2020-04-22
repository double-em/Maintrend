# sudo rm -rf /home/marianne/container-data/LSTMdata/*
path=/home/$(whoami)/container-data/LSTMdata/$(date '+%F@%H;%M;%S')
mkdir -p $path
docker build -t trainer:latest .
docker run -it --rm --gpus all -v $path:/logs trainer:latest