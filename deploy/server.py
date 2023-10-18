import sys
import os

import tensorflow as tf
import tensorflow_recommenders as tfrs

from flask import Flask, request

# Environment variables
PORT = os.environ["AIP_HTTP_PORT"]
HEALTH_ROUTE = os.environ["AIP_HEALTH_ROUTE"]
PREDICT_ROUTE = os.environ["AIP_PREDICT_ROUTE"]
DEFAULT_K = os.environ["DEFAULT_K"]

SCANN_MODEL_DIR = os.environ["SCANN_MODEL_DIR"]
scann_loaded = tf.keras.models.load_model(SCANN_MODEL_DIR)

# Configure Flask app
app = Flask(__name__)


@app.route(HEALTH_ROUTE)
def health():
    return "OK"


@app.post(PREDICT_ROUTE)
def index():
    body = request.json

    instances = body["instances"]
    parameters = body.get("parameters", {})

    k = int(parameters.get("k", DEFAULT_K))

    predictions = [
        _predict(
            user_id=instance["user_id"],
            exclusions=instance.get("exclusions", []),
            k=k,
        )
        for instance in instances
    ]

    return {"predictions": predictions}


def _predict(user_id, exclusions=[], k=DEFAULT_K):
    user_ids = tf.constant([user_id])
    k = tf.constant(k)

    if len(exclusions) > 0:
        exclusions = tf.constant([exclusions])
        k_scores, k_predictions = scann_loaded.query_with_exclusions(
            user_ids, exclusions, k
        )
    else:
        k_scores, k_predictions = scann_loaded.call(user_ids, k)

    k_predictions_json = k_predictions.numpy().tolist()
    k_scores_json = k_scores.numpy().tolist()

    k_encoded_predictions_json = [
        [pred.decode("utf-8") for pred in pred_list] for pred_list in k_predictions_json
    ]

    return {
        "movie_titles": k_encoded_predictions_json[0],
        "movie_scores": k_scores_json[0],
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
