FROM tensorflow/tensorflow:2.13.0

WORKDIR /serve

COPY models/scann_100k scann_index

# Install server dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Prediction server source code
COPY server.py .

ENV SCANN_MODEL_DIR="scann_index"
ENV DEFAULT_K="10"

ENTRYPOINT ["python3", "server.py"]
