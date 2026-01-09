"""
Configuration MLFlow pour le projet de credit scoring
"""
import os
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).parent.parent
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR}/mlruns/mlflow.db"
MLFLOW_ARTIFACT_ROOT = str(BASE_DIR / "mlartifacts")

# Configuration des expériences
EXPERIMENT_NAME = "credit-scoring"
MODEL_REGISTRY_NAME = "credit-scoring-models"

# Configuration du serveur MLFlow
MLFLOW_SERVER_CONFIG = {
    "backend_store_uri": MLFLOW_TRACKING_URI,
    "default_artifact_root": MLFLOW_ARTIFACT_ROOT,
    "host": "127.0.0.1",
    "port": 5000,
    "workers": 1
}

# Tags par défaut pour les expériences
DEFAULT_TAGS = {
    "project": "credit-scoring",
    "company": "pret-a-depenser",
    "task": "binary-classification",
    "domain": "finance"
}

# Métriques importantes à tracker
IMPORTANT_METRICS = [
    "accuracy",
    "precision",
    "recall", 
    "f1_score",
    "auc_roc",
    "auc_pr",
    "business_cost",
    "optimal_threshold"
]

# Paramètres importants à tracker
IMPORTANT_PARAMS = [
    "model_type",
    "random_state",
    "test_size",
    "feature_selection",
    "class_balance_method",
    "cost_fn_weight"
]
