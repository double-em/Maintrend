path_code=$(pwd)/serving-api/
path_util=$(pwd)/util/

if [ "$(docker images -q serving-api:latest)" == "" ]; then
    docker build -t serving-api:latest -f serving-api/Dockerfile .
fi

docker run -it --rm --gpus all \
-e API_BASE_URL=$API_BASE_URL \
-e API_F=$API_F \
--name serving-api \
--network maintrend-net \
serving-api:latest