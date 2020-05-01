path=/home/$(whoami)/container-data/
path_logs=$path/logs/
path_models=$path/models/
mkdir -p $path_logs
mkdir -p $path_models

# path_logs=/run/user/1000/gvfs/'google-drive:host=edu.ucl.dk,user=mike2860'/ml-exp/logs/
# path_models=/run/user/1000/gvfs/'google-drive:host=edu.ucl.dk,user=mike2860'/ml-exp/data/

docker build -t trainer:latest .
docker run -it --rm --gpus all -v $path_logs:/logs -v $path_models:/models trainer:latest
# -u $(id -u):$(id -g)