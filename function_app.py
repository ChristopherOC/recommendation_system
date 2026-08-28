import azure.functions as func
from azure.storage.blob import BlobServiceClient

import pickle
import os
import json
import logging
import threading
import numpy as np


app = func.FunctionApp(
    http_auth_level=func.AuthLevel.ANONYMOUS
)


# ============================================================
# CONFIGURATION
# ============================================================

STORAGE_CONNECTION_STRING = os.getenv(
    "AZURE_STORAGE_CONNECTION_STRING"
)

MODEL_CONTAINER = os.getenv(
    "MODEL_CONTAINER"
)

SVD_BLOB_NAME = os.getenv(
    "SVD_BLOB_NAME",
    "svd_model_light.pkl",
)

SEEN_BLOB_NAME = os.getenv(
    "SEEN_BLOB_NAME",
    "seen_articles.pkl",
)

ARTICLE_IDS_BLOB_NAME = os.getenv(
    "ARTICLE_IDS_BLOB_NAME",
    "article_ids.pkl",
)

LOCAL_MODEL_DIR = "/tmp/models"


# ============================================================
# VARIABLES GLOBALES
# ============================================================

svd_model = None
seen_by_user = None
article_ids = None

model_load_lock = threading.Lock()


# ============================================================
# TÉLÉCHARGEMENT BLOB
# ============================================================

def download_blob_if_needed(
    blob_service_client,
    blob_name,
    local_filename,
):

    os.makedirs(
        LOCAL_MODEL_DIR,
        exist_ok=True,
    )

    local_path = os.path.join(
        LOCAL_MODEL_DIR,
        local_filename,
    )

    if os.path.exists(local_path):
        logging.info(
            "Fichier déjà présent localement : %s",
            local_path,
        )
        return local_path

    logging.info(
        "Téléchargement du blob '%s' depuis '%s'",
        blob_name,
        MODEL_CONTAINER,
    )

    blob_client = blob_service_client.get_blob_client(
        container=MODEL_CONTAINER,
        blob=blob_name,
    )

    with open(local_path, "wb") as f:
        stream = blob_client.download_blob()
        stream.readinto(f)

    logging.info(
        "Téléchargement terminé : %s",
        local_path,
    )

    return local_path


# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================

def load_models():

    global svd_model
    global seen_by_user
    global article_ids

    if (
        svd_model is not None
        and seen_by_user is not None
        and article_ids is not None
    ):
        return

    with model_load_lock:

        if (
            svd_model is not None
            and seen_by_user is not None
            and article_ids is not None
        ):
            return

        if not STORAGE_CONNECTION_STRING:
            raise RuntimeError(
                "AZURE_STORAGE_CONNECTION_STRING manquant"
            )

        if not MODEL_CONTAINER:
            raise RuntimeError(
                "MODEL_CONTAINER manquant"
            )

        logging.info(
            "Début du chargement des modèles"
        )

        blob_service_client = (
            BlobServiceClient.from_connection_string(
                STORAGE_CONNECTION_STRING
            )
        )

        svd_path = download_blob_if_needed(
            blob_service_client,
            SVD_BLOB_NAME,
            "svd_model_light.pkl",
        )

        seen_path = download_blob_if_needed(
            blob_service_client,
            SEEN_BLOB_NAME,
            "seen_articles.pkl",
        )

        article_ids_path = download_blob_if_needed(
            blob_service_client,
            ARTICLE_IDS_BLOB_NAME,
            "article_ids.pkl",
        )

        logging.info(
            "Chargement du modèle SVD léger"
        )

        with open(svd_path, "rb") as f:
            loaded_svd = pickle.load(f)

        logging.info(
            "Chargement des articles vus"
        )

        with open(seen_path, "rb") as f:
            loaded_seen = pickle.load(f)

        logging.info(
            "Chargement des IDs articles"
        )

        with open(article_ids_path, "rb") as f:
            loaded_article_ids = pickle.load(f)

        svd_model = loaded_svd
        seen_by_user = loaded_seen
        article_ids = loaded_article_ids

        logging.info(
            "Tous les modèles sont chargés"
        )


# ============================================================
# PRÉDICTION SVD
# ============================================================

def predict_svd(
    user_id,
    article_id,
):

    mean = svd_model["global_mean"]

    raw2inner_user = svd_model[
        "raw2inner_user"
    ]

    raw2inner_item = svd_model[
        "raw2inner_item"
    ]

    uid = raw2inner_user.get(
        user_id
    )

    iid = raw2inner_item.get(
        article_id
    )

    score = mean

    if uid is not None:
        score += float(
            svd_model["bu"][uid]
        )

    if iid is not None:
        score += float(
            svd_model["bi"][iid]
        )

    if (
        uid is not None
        and iid is not None
    ):

        score += float(
            np.dot(
                svd_model["pu"][uid],
                svd_model["qi"][iid],
            )
        )


    return score

# ============================================================
# RECOMMANDATION
# ============================================================

def recommend_svd(user_id, n=5):

    load_models()

    seen = seen_by_user.get(
        user_id,
        set(),
    )

    candidates = [
        article_id
        for article_id in article_ids
        if article_id not in seen
    ]

    predictions = [
        (
            int(article_id),
            predict_svd(
                user_id,
                article_id,
            ),
        )
        for article_id in candidates
    ]

    predictions.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return predictions[:n]


# ============================================================
# ROUTE RECOMMEND
# ============================================================

@app.route(
    route="recommend",
    methods=["GET"],
)
def recommend(
    req: func.HttpRequest,
) -> func.HttpResponse:

    logging.info(
        "Requête de recommandation reçue"
    )

    user_id_raw = req.params.get(
        "user_id"
    )

    n_raw = req.params.get(
        "n",
        "5",
    )

    if user_id_raw is None:

        return func.HttpResponse(
            json.dumps(
                {
                    "error": (
                        "Le paramètre 'user_id' "
                        "est requis"
                    )
                }
            ),
            status_code=400,
            mimetype="application/json",
        )

    try:

        user_id = int(
            user_id_raw
        )

        n = int(
            n_raw
        )

    except ValueError:

        return func.HttpResponse(
            json.dumps(
                {
                    "error": (
                        "'user_id' et 'n' "
                        "doivent être des entiers"
                    )
                }
            ),
            status_code=400,
            mimetype="application/json",
        )

    if n <= 0 or n > 100:

        return func.HttpResponse(
            json.dumps(
                {
                    "error": (
                        "'n' doit être compris "
                        "entre 1 et 100"
                    )
                }
            ),
            status_code=400,
            mimetype="application/json",
        )

    try:

        recommendations = recommend_svd(
            user_id,
            n,
        )

        result = {
            "user_id": user_id,
            "recommendations": [
                {
                    "article_id": article_id,
                    "score": round(
                        score,
                        4,
                    ),
                }
                for article_id, score
                in recommendations
            ],
        }

        return func.HttpResponse(
            json.dumps(
                result,
                ensure_ascii=False,
            ),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:

        logging.exception(
            "Erreur pendant la recommandation"
        )

        return func.HttpResponse(
            json.dumps(
                {
                    "error": (
                        "Erreur interne pendant "
                        "la recommandation"
                    ),
                    "details": str(e),
                },
                ensure_ascii=False,
            ),
            status_code=500,
            mimetype="application/json",
        )


# ============================================================
# ROUTE HEALTH
# ============================================================

@app.route(
    route="health",
    methods=["GET"],
)
def health(
    req: func.HttpRequest,
) -> func.HttpResponse:

    result = {
        "status": "ok",
        "function_loaded": True,
        "storage_connection_configured": bool(
            STORAGE_CONNECTION_STRING
        ),
        "model_container_configured": bool(
            MODEL_CONTAINER
        ),
        "models_loaded": (
            svd_model is not None
            and seen_by_user is not None
            and article_ids is not None
        ),
        "model_container": MODEL_CONTAINER,
        "svd_blob": SVD_BLOB_NAME,
    }

    return func.HttpResponse(
        json.dumps(
            result,
            ensure_ascii=False,
        ),
        status_code=200,
        mimetype="application/json",
    )
