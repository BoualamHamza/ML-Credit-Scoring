"""
Utilitaires MLFlow pour le tracking des expérimentations
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import joblib
from pathlib import Path
import json

from config.mlflow_config import (
    EXPERIMENT_NAME, MODEL_REGISTRY_NAME, DEFAULT_TAGS,
    IMPORTANT_METRICS, IMPORTANT_PARAMS
)


class MLFlowTracker:
    """Classe pour gérer le tracking MLFlow des expérimentations"""
    
    def __init__(self, experiment_name: str = EXPERIMENT_NAME):
        """Initialise le tracker MLFlow"""
        self.experiment_name = experiment_name
        self.setup_experiment()
    
    def setup_experiment(self):
        """Configure l'expérience MLFlow"""
        try:
            # Créer ou récupérer l'expérience
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    tags=DEFAULT_TAGS
                )
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_id=experiment_id)
            print(f"✅ Expérience MLFlow configurée: {self.experiment_name}")
            
        except Exception as e:
            print(f"❌ Erreur lors de la configuration MLFlow: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Démarre une nouvelle run MLFlow"""
        run_tags = DEFAULT_TAGS.copy()
        if tags:
            run_tags.update(tags)
        
        return mlflow.start_run(run_name=run_name, tags=run_tags)
    
    def log_model_params(self, params: Dict[str, Any]):
        """Log les paramètres du modèle"""
        # Filtrer les paramètres importants
        important_params = {k: v for k, v in params.items() 
                          if k in IMPORTANT_PARAMS or 'model' in k.lower()}
        mlflow.log_params(important_params)
    
    def log_data_info(self, X_train: pd.DataFrame, y_train: pd.Series, 
                     X_test: pd.DataFrame, y_test: pd.Series):
        """Log les informations sur les données"""
        data_info = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_count": X_train.shape[1],
            "train_positive_rate": y_train.mean(),
            "test_positive_rate": y_test.mean(),
            "class_imbalance_ratio": (1 - y_train.mean()) / y_train.mean()
        }
        mlflow.log_params(data_info)
    
    def log_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                   y_pred_proba: np.ndarray, business_cost: float = None,
                   optimal_threshold: float = None):
        """Log les métriques de performance"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "auc_pr": average_precision_score(y_true, y_pred_proba)
        }
        
        if business_cost is not None:
            metrics["business_cost"] = business_cost
        
        if optimal_threshold is not None:
            metrics["optimal_threshold"] = optimal_threshold
        
        # Log de la matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        })
        
        mlflow.log_metrics(metrics)
        return metrics
    
    def log_model(self, model: Any, model_name: str, 
                 model_type: str = "sklearn", 
                 feature_names: List[str] = None,
                 model_signature: Any = None):
        """Log le modèle dans MLFlow"""
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(
                    model, 
                    model_name,
                    signature=model_signature
                )
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(
                    model,
                    model_name,
                    signature=model_signature
                )
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(
                    model,
                    model_name,
                    signature=model_signature
                )
            else:
                # Modèle générique
                mlflow.sklearn.log_model(
                    model,
                    model_name,
                    signature=model_signature
                )
            
            print(f"✅ Modèle {model_name} loggé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors du logging du modèle: {e}")
            raise
    
    def log_artifacts(self, artifacts_dir: str, artifact_path: str = None):
        """Log des artifacts (graphiques, fichiers, etc.)"""
        mlflow.log_artifacts(artifacts_dir, artifact_path)
    
    def log_figure(self, figure, artifact_name: str):
        """Log une figure matplotlib/plotly"""
        mlflow.log_figure(figure, artifact_name)
    
    def calculate_business_cost(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             fn_cost: float = 10.0, fp_cost: float = 1.0):
        """Calcule le coût métier (FN coûte 10x plus que FP)"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total_cost = (fn * fn_cost) + (fp * fp_cost)
        return total_cost
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              fn_cost: float = 10.0, fp_cost: float = 1.0):
        """Trouve le seuil optimal basé sur le coût métier"""
        from sklearn.metrics import precision_recall_curve
        
        thresholds = np.linspace(0, 1, 100)
        costs = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            cost = self.calculate_business_cost(y_true, y_pred_thresh, fn_cost, fp_cost)
            costs.append(cost)
        
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold, costs[optimal_idx]
    
    def register_model(self, model_name: str, model_version: str = None,
                      description: str = None, tags: Dict[str, str] = None):
        """Enregistre le modèle dans le registry"""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Créer le modèle dans le registry s'il n'existe pas
            try:
                client.create_registered_model(MODEL_REGISTRY_NAME)
            except:
                pass  # Le modèle existe déjà
            
            # Enregistrer la version
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
            
            model_version = client.create_model_version(
                name=MODEL_REGISTRY_NAME,
                source=model_uri,
                description=description,
                tags=tags
            )
            
            print(f"✅ Modèle enregistré: {MODEL_REGISTRY_NAME}/{model_version.version}")
            return model_version
            
        except Exception as e:
            print(f"❌ Erreur lors de l'enregistrement du modèle: {e}")
            raise


def get_best_model_by_metric(metric_name: str = "business_cost", 
                            ascending: bool = True) -> Optional[Dict]:
    """Récupère le meilleur modèle selon une métrique"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is None:
            return None
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        
        if runs:
            best_run = runs[0]
            return {
                "run_id": best_run.info.run_id,
                "metric_value": best_run.data.metrics.get(metric_name),
                "model_uri": f"runs:/{best_run.info.run_id}/model"
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération du meilleur modèle: {e}")
        return None
