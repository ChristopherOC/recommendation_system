import azure.functions as func
from azure.storage.blob import BlobServiceClient

import pickle
import os
import json
import logging


# ============================================================
# AZURE FUNCTION
# ============================================================

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


# ============================================================
# CONFIGURATION BLOB STORAGE
# ============================================================

# À définir dans Azure > Function App > Environment variables
STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
MODEL_CONTAINER = os.environ["MODEL_CONTAINER"]

# Tu peux laisser ces noms tels quels si tes blobs portent ces noms.
SVD_BLOB_NAME = os.getenv("SVD_BLOB_NAME", "svd_model.pkl")
SEEN_BLOB_NAME = os.getenv("SEEN_BLOB_NAME", "seen_articles.pkl")
ARTICLE_IDS_BLOB_NAME = os.getenv("ARTICLE_IDS_BLOB_NAME", "article_ids.pkl")


# Les fichiers téléchargés seront temporairement stockés ici.
# /tmp est accessible en écriture dans Azure Functions Linux.
LOCAL_MODEL_DIR = "/tmp/models"

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)


# ============================================================
# TÉLÉCHARGEMENT DES MODÈLES
# ============================================================

blob_service_client = BlobServiceClient.from_connection_string(
    STORAGE_CONNECTION_STRING
)


def download_blob_if_needed(blob_name: str, local_filename: str) -> str:
    """
    Télécharge un blob dans /tmp uniquement s'il n'est pas déjà présent
    sur l'instance Azure Functions.
    """

    local_path = os.path.join(LOCAL_MODEL_DIR, local_filename)

    if os.path.exists(local_path):
        logging.info("Fichier déjà présent localement : %s", local_path)
        return local_path

    logging.info(
        "Téléchargement du blob '%s' depuis le container '%s'",
        blob_name,
        MODEL_CONTAINER,
    )

    blob_client = blob_service_client.get_blob_client(
        container=MODEL_CONTAINER,
        blob=blob_name,
    )

    with open(local_path, "wb") as file:
        stream = blob_client.download_blob()
        stream.readinto(file)

    logging.info("Téléchargement terminé : %s", local_path)

    return local_path


# ============================================================
# CHARGEMENT DES ARTEFACTS
# ============================================================

try:
    svd_path = download_blob_if_needed(
        SVD_BLOB_NAME,
        "svd_model.pkl",
    )

    seen_path = download_blob_if_needed(
        SEEN_BLOB_NAME,
        "seen_articles.pkl",
    )

    article_ids_path = download_blob_if_needed(
        ARTICLE_IDS_BLOB_NAME,
        "article_ids.pkl",
    )

    with open(svd_path, "rb") as f:
        algo = pickle.load(f)

    with open(seen_path, "rb") as f:
        seen_by_user = pickle.load(f)

    with open(article_ids_path, "rb") as f:
        article_ids = pickle.load(f)

    logging.info(
        "Modèle SVD et données chargés avec succès (%d articles candidats)",
        len(article_ids),
    )

except Exception:
    logging.exception(
        "Erreur pendant le téléchargement ou le chargement des modèles"
    )
    raise


# ============================================================
# RECOMMANDATION
# ============================================================

def recommend_surprise(user_id, n=5):

    seen = seen_by_user.get(user_id, set())

    candidates = [
        article_id
        for article_id in article_ids
        if article_id not in seen
    ]

    predictions = [
        (
            int(article_id),
            float(algo.predict(user_id, article_id).est),
        )
        for article_id in candidates
    ]

    predictions.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return predictions[:n]


# ============================================================
# ROUTE HTTP
# ============================================================

@app.route(route="recommend", methods=["GET"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:

    logging.info("Requête de recommandation reçue")

    user_id_raw = req.params.get("user_id")
    n_raw = req.params.get("n", "5")

    if user_id_raw is None:
        return func.HttpResponse(
            json.dumps(
                {
                    "error": "Le paramètre 'user_id' est requis"
                },
                ensure_ascii=False,
            ),
            status_code=400,
            mimetype="application/json",
        )

    try:
        user_id = int(user_id_raw)
        n = int(n_raw)

    except ValueError:
        return func.HttpResponse(
            json.dumps(
                {
                    "error": "'user_id' et 'n' doivent être des entiers"
                },
                ensure_ascii=False,
            ),
            status_code=400,
            mimetype="application/json",
        )

    if n <= 0 or n > 100:
        return func.HttpResponse(
            json.dumps(
                {
                    "error": "'n' doit être compris entre 1 et 100"
                },
                ensure_ascii=False,
            ),
            status_code=400,
            mimetype="application/json",
        )

    try:

        recommendations = recommend_surprise(
            user_id,
            n,
        )

        result = {
            "user_id": user_id,
            "recommendations": [
                {
                    "article_id": article_id,
                    "score": round(score, 4),
                }
                for article_id, score in recommendations
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
            "Erreur pendant la génération des recommandations"
        )

        return func.HttpResponse(
            json.dumps(
                {
                    "error": "Erreur interne pendant la recommandation"
                },
                ensure_ascii=False,
            ),
            status_code=500,
            mimetype="application/json",
        )
