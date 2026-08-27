import azure.functions as func
from azure.storage.blob import BlobServiceClient

import pickle
import os
import json
import logging
import threading


# ============================================================
# AZURE FUNCTION
# ============================================================

app = func.FunctionApp(
    http_auth_level=func.AuthLevel.ANONYMOUS
)


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT :
# On utilise os.getenv() et non os.environ[] pour éviter de faire
# planter l'import de function_app.py si une variable manque.
#
# Les variables seront vérifiées seulement au moment où une
# requête aura besoin des modèles.

STORAGE_CONNECTION_STRING = os.getenv(
    "AZURE_STORAGE_CONNECTION_STRING"
)

MODEL_CONTAINER = os.getenv(
    "MODEL_CONTAINER"
)


# Noms des blobs.
# Ils peuvent être surchargés avec des variables d'environnement.

SVD_BLOB_NAME = os.getenv(
    "SVD_BLOB_NAME",
    "svd_model.pkl",
)

SEEN_BLOB_NAME = os.getenv(
    "SEEN_BLOB_NAME",
    "seen_articles.pkl",
)

ARTICLE_IDS_BLOB_NAME = os.getenv(
    "ARTICLE_IDS_BLOB_NAME",
    "article_ids.pkl",
)


# ============================================================
# STOCKAGE TEMPORAIRE LOCAL
# ============================================================

# Azure Functions Linux autorise l'écriture dans /tmp.
# Les modèles seront conservés ici tant que l'instance Azure
# reste active.

LOCAL_MODEL_DIR = "/tmp/models"


# ============================================================
# VARIABLES GLOBALES DES MODÈLES
# ============================================================

# Les modèles ne sont PAS chargés au démarrage de la Function.
# Ils seront chargés uniquement lors de la première requête.

algo = None
seen_by_user = None
article_ids = None


# Empêche deux requêtes simultanées de charger les modèles
# en même temps lors du premier appel.

model_load_lock = threading.Lock()


# ============================================================
# TÉLÉCHARGEMENT D'UN BLOB
# ============================================================

def download_blob_if_needed(
    blob_service_client,
    blob_name: str,
    local_filename: str,
) -> str:

    os.makedirs(
        LOCAL_MODEL_DIR,
        exist_ok=True,
    )

    local_path = os.path.join(
        LOCAL_MODEL_DIR,
        local_filename,
    )

    # Si le fichier est déjà présent sur l'instance Azure,
    # inutile de le retélécharger.

    if os.path.exists(local_path):

        logging.info(
            "Fichier déjà présent localement : %s",
            local_path,
        )

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


    logging.info(
        "Téléchargement terminé : %s",
        local_path,
    )


    return local_path


# ============================================================
# CHARGEMENT LAZY DES MODÈLES
# ============================================================

def load_models():

    global algo
    global seen_by_user
    global article_ids


    # Si tout est déjà chargé, on retourne immédiatement.

    if (
        algo is not None
        and seen_by_user is not None
        and article_ids is not None
    ):
        return


    # Empêche plusieurs chargements simultanés.

    with model_load_lock:


        # Deuxième vérification après acquisition du lock.
        # Une autre requête a peut-être terminé le chargement
        # pendant qu'on attendait.

        if (
            algo is not None
            and seen_by_user is not None
            and article_ids is not None
        ):
            return


        logging.info(
            "Début du chargement des modèles..."
        )


        # ====================================================
        # VÉRIFICATION DE LA CONFIGURATION
        # ====================================================

        if not STORAGE_CONNECTION_STRING:

            raise RuntimeError(
                "La variable d'environnement "
                "'AZURE_STORAGE_CONNECTION_STRING' "
                "n'est pas définie."
            )


        if not MODEL_CONTAINER:

            raise RuntimeError(
                "La variable d'environnement "
                "'MODEL_CONTAINER' "
                "n'est pas définie."
            )


        # ====================================================
        # CONNEXION À AZURE BLOB STORAGE
        # ====================================================

        logging.info(
            "Connexion à Azure Blob Storage..."
        )


        blob_service_client = (
            BlobServiceClient.from_connection_string(
                STORAGE_CONNECTION_STRING
            )
        )


        # ====================================================
        # TÉLÉCHARGEMENT DES FICHIERS
        # ====================================================

        svd_path = download_blob_if_needed(
            blob_service_client,
            SVD_BLOB_NAME,
            "svd_model.pkl",
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


        # ====================================================
        # CHARGEMENT DES PICKLES
        # ====================================================

        logging.info(
            "Chargement du modèle SVD..."
        )


        with open(
            svd_path,
            "rb",
        ) as f:

            loaded_algo = pickle.load(f)


        logging.info(
            "Chargement des articles déjà vus..."
        )


        with open(
            seen_path,
            "rb",
        ) as f:

            loaded_seen_by_user = pickle.load(f)


        logging.info(
            "Chargement de la liste des articles..."
        )


        with open(
            article_ids_path,
            "rb",
        ) as f:

            loaded_article_ids = pickle.load(f)


        # On affecte les variables globales seulement une fois
        # que les trois chargements ont réussi.

        algo = loaded_algo
        seen_by_user = loaded_seen_by_user
        article_ids = loaded_article_ids


        logging.info(
            "Modèles chargés avec succès : %d articles candidats",
            len(article_ids),
        )


# ============================================================
# RECOMMANDATION
# ============================================================

def recommend_surprise(
    user_id,
    n=5,
):

    # Sécurité supplémentaire :
    # si la fonction est appelée directement, les modèles
    # seront tout de même chargés.

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
            float(
                algo.predict(
                    user_id,
                    article_id,
                ).est
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
# ROUTE HTTP
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


    # ========================================================
    # PARAMÈTRES
    # ========================================================

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
                },
                ensure_ascii=False,
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
                    "error": (
                        "'n' doit être compris "
                        "entre 1 et 100"
                    )
                },
                ensure_ascii=False,
            ),
            status_code=400,
            mimetype="application/json",
        )


    # ========================================================
    # RECOMMANDATION
    # ========================================================

    try:

        logging.info(
            "Vérification / chargement des modèles"
        )


        load_models()


        logging.info(
            "Génération des recommandations "
            "pour user_id=%d, n=%d",
            user_id,
            n,
        )


        recommendations = recommend_surprise(
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


        logging.info(
            "Recommandation terminée avec succès "
            "pour user_id=%d",
            user_id,
        )


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
            "Erreur pendant le chargement des modèles "
            "ou la génération des recommandations"
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
# ROUTE DE DIAGNOSTIC
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
            algo is not None
            and seen_by_user is not None
            and article_ids is not None
        ),

        "model_container": MODEL_CONTAINER,

    }


    return func.HttpResponse(
        json.dumps(
            result,
            ensure_ascii=False,
        ),
        status_code=200,
        mimetype="application/json",
    )
