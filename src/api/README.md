# Credit Scoring API & Streamlit Interface

Ce dossier contient l'API FastAPI et l'interface Streamlit pour le système de scoring de crédit.

## Structure

- `api.py` : API FastAPI avec endpoints pour les prédictions
- `streamlit_app.py` : Interface utilisateur Streamlit

## Prérequis

Assurez-vous que les fichiers suivants existent :
- `src/models/lgbm_best_model.pkl` : Modèle LightGBM entraîné
- `src/dataset/features_train.pkl` : Dataset complet avec features
- `src/dataset/feature_names.json` : Liste des noms de features

## Utilisation

### 1. Démarrer l'API FastAPI

```bash
# Option 1 : Avec uvicorn directement
uvicorn src.api.api:app --reload --host 0.0.0.0 --port 8000

# Option 2 : Exécuter le fichier Python
python -m src.api.api
```

L'API sera accessible sur : `http://localhost:8000`

Documentation interactive disponible sur : `http://localhost:8000/docs`

### 2. Démarrer l'interface Streamlit

Dans un autre terminal :

```bash
streamlit run src/api/streamlit_app.py
```

L'interface sera accessible sur : `http://localhost:8501`

## Endpoints API

### GET /health
Vérifie l'état de l'API et si le modèle est chargé.

**Réponse :**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "lightgbm"
}
```

### GET /model/info
Retourne les informations sur le modèle.

**Réponse :**
```json
{
  "model_type": "lightgbm",
  "n_features": 763,
  "optimal_threshold": 0.475,
  "feature_names": [...],
  "model_loaded": true
}
```

### POST /predict/client_id
Fait une prédiction pour un ID client spécifique.

**Requête :**
```json
{
  "client_id": 100001
}
```

**Réponse :**
```json
{
  "client_id": 100001,
  "probability": 0.234,
  "prediction": 0,
  "threshold": 0.475,
  "recommendation": "✅ Risque de défaut faible - Prêt recommandé"
}
```

## Tests

Exécuter les tests unitaires :

```bash
pytest tests/test_api.py -v
```

## Notes

- Le seuil optimal utilisé est **0.475** (déterminé lors de l'entraînement)
- Les prédictions avec probabilité >= 0.475 sont classées comme "risque élevé" (prediction = 1)
- L'API charge le modèle depuis le fichier local (`src/models/lgbm_best_model.pkl`)
- Le dataset complet est chargé en mémoire pour permettre la recherche rapide par ID client

## Déploiement Azure

L'API peut être déployée sur Azure Web App Services via GitHub Actions.

### Configuration requise

1. **Secrets GitHub** à configurer :
   - `AZURE_CREDENTIALS` : Service Principal credentials (JSON)
   - `AZURE_WEBAPP_NAME` : Nom de l'Azure Web App
   - `AZURE_RESOURCE_GROUP` : Nom du resource group (optionnel)

2. **Azure Web App Configuration** :
   - Runtime : Python 3.11
   - Startup command : `bash startup.sh`
   - Port : Azure définit automatiquement la variable d'environnement `PORT`

### Déploiement automatique

Le pipeline CI/CD se déclenche automatiquement sur push vers `main` ou `master` :
- Exécute les tests
- Déploie vers Azure si les tests passent

Voir `.github/workflows/azure-deploy.yml` pour plus de détails.
