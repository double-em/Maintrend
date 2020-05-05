if [ "$(docker images -q serving-api:latest)" == "" ]; then
    docker build -t serving-api:latest -f serving-api/Dockerfile .
fi

docker run -it --rm --gpus all \
-e API_BASE_URL=$API_BASE_URL \
-e API_CHANNEL=$API_CHANNEL \
-e API_F=$API_F \
-e API_KEY=$API_KEY \
--name serving-api \
serving-api:latest