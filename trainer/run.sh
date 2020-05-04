path=$(pwd)/container-data/
path_logs=$path/logs/
path_models=$path/models/
path_code=$(pwd)/trainer/
path_util=$(pwd)/util/

mkdir -p $path_logs
mkdir -p $path_models

if [ "$(docker images -q maintrend-trainer:latest)" == "" ]; then
    docker build -t maintrend-trainer:latest -f trainer/Dockerfile .
fi

docker run -it --rm --gpus all \
-v $path_code:/app \
-v $path_util:/app/util \
-v $path_logs:/app/logs \
-v $path_models:/app/models \
-e API_BASE_URL=$API_BASE_URL \
-e API_CHANNEL=$API_CHANNEL \
-e API_F=$API_F \
-e API_KEY=$API_KEY \
maintrend-trainer:latest