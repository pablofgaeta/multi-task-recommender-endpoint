export AIP_HTTP_PORT=80
export AIP_HEALTH_ROUTE=/health
export AIP_PREDICT_ROUTE=/predict

docker stop scann-index
docker rm scann-index
docker build -t scann_index deploy/
docker run -d --name scann-index \
    -p 80:80 \
    -e AIP_HTTP_PORT -e AIP_HEALTH_ROUTE -e AIP_PREDICT_ROUTE \
    scann_index
# Sleep to allow the server to start up
sleep 5
curl -X POST -H "Content-Type: application/json" -d @deploy/request.json http://127.0.0.1:80/predict