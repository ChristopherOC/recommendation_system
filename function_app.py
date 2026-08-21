import azure.functions as func
import pickle
import os
import json
import logging

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Chargement des artefacts une seule fois, au démarrage de la fonction
# (réutilisé pour toutes les requêtes tant que l'instance reste "chaude")
with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "rb") as f:
    algo = pickle.load(f)

with open(os.path.join(MODEL_DIR, "seen_articles.pkl"), "rb") as f:
    seen_by_user = pickle.load(f)

with open(os.path.join(MODEL_DIR, "article_ids.pkl"), "rb") as f:
    article_ids = pickle.load(f)

logging.info("Modèle SVD et données chargés (%d articles candidats)", len(article_ids))


def recommend_surprise(user_id, n=5):
    """Reprend exactement la logique de la fonction du notebook."""
    seen = seen_by_user.get(user_id, set())
    candidates = [a for a in article_ids if a not in seen]

    predictions = [(int(a), float(algo.predict(user_id, a).est)) for a in candidates]
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:n]


@app.route(route="recommend", methods=["GET"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Requête de recommandation reçue")

    user_id_raw = req.params.get("user_id")
    n_raw = req.params.get("n", "5")

    if user_id_raw is None:
        return func.HttpResponse(
            json.dumps({"error": "Le paramètre 'user_id' est requis"}, ensure_ascii=False),
            status_code=400,
            mimetype="application/json"
        )

    try:
        user_id = int(user_id_raw)
        n = int(n_raw)
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "'user_id' et 'n' doivent être des entiers"}, ensure_ascii=False),
            status_code=400,
            mimetype="application/json"
        )

    recos = recommend_surprise(user_id, n)

    result = {
        "user_id": user_id,
        "recommendations": [
            {"article_id": article_id, "score": round(score, 4)}
            for article_id, score in recos
        ]
    }

    return func.HttpResponse(
        json.dumps(result, ensure_ascii=False),
        status_code=200,
        mimetype="application/json"
    )
