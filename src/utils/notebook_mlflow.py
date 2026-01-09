"""
Utilitaires MLFlow pour les notebooks Jupyter
Supports cross-validation, Optuna hyperparameter tuning, and model tracking.
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# Business cost constants
FN_COST = 10.0  # False Negative cost (missed default)
FP_COST = 1.0   # False Positive cost (good client rejected)

class NotebookMLFlow:
    """Classe pour faciliter l'utilisation de MLFlow dans les notebooks"""
    
    def __init__(self, experiment_name="credit-scoring-notebook", tracking_uri=None):
        """Initialise MLFlow pour les notebooks"""
        if tracking_uri is None:
            # Use absolute path to project root mlruns (works regardless of notebook location)
            project_root = Path(__file__).resolve().parent.parent.parent
            tracking_uri = f"sqlite:///{project_root}/mlruns/mlflow.db"
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Créer ou récupérer l'expérience
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # Force artifacts to be stored in project-root `mlartifacts/` (avoid creating `notebooks/mlruns/`)
                artifact_root = (project_root / "mlartifacts").resolve()
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_root.as_uri(),
                    tags={
                        "project": "credit-scoring",
                        "notebook": "true",
                        "domain": "finance"
                    }
                )
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_id=experiment_id)
            self.experiment_id = experiment_id
            print(f"✅ MLFlow configuré - Expérience: {experiment_name}")
            
        except Exception as e:
            print(f"❌ Erreur MLFlow: {e}")
            raise
    
    def log_experiment(self, model, model_name, X_train, X_test, y_train, y_test, 
                     params=None, tags=None, create_plots=True):
        """Log une expérimentation complète"""
        
        # Tags par défaut
        run_tags = {
            "model_type": model_name,
            "domain": "credit-scoring",
            "notebook": "true"
        }
        if tags:
            run_tags.update(tags)
        
        with mlflow.start_run(run_name=f"{model_name}_{np.random.randint(1000, 9999)}", tags=run_tags):
            
            # Log des paramètres
            if params:
                mlflow.log_params(params)
            
            # Informations sur les données
            data_info = {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features_count": X_train.shape[1],
                "train_positive_rate": y_train.mean(),
                "test_positive_rate": y_test.mean(),
                "class_imbalance_ratio": (1 - y_train.mean()) / y_train.mean()
            }
            mlflow.log_params(data_info)
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Métriques
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            mlflow.log_metrics(metrics)
            
            # Log du modèle
            mlflow.sklearn.log_model(
                model, 
                "model",
                signature=mlflow.models.infer_signature(X_test, y_pred_proba)
            )
            
            # Visualisations
            if create_plots:
                self._create_plots(y_test, y_pred, y_pred_proba, model_name)
            
            print(f"✅ {model_name} - Accuracy: {metrics['accuracy']:.3f}, AUC: {metrics['auc_roc']:.3f}, Business Cost: {metrics['business_cost']:.1f}")
            
            return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calcule toutes les métriques importantes"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_true, y_pred_proba)
        }
        
        # Coût métier (FN coûte 10x plus que FP)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        business_cost = (fn * 10) + (fp * 1)
        metrics["business_cost"] = business_cost
        
        # Seuil optimal
        optimal_threshold, optimal_cost = self._find_optimal_threshold(y_true, y_pred_proba)
        metrics.update({
            "optimal_threshold": optimal_threshold,
            "optimal_business_cost": optimal_cost
        })
        
        return metrics
    
    def _find_optimal_threshold(self, y_true, y_pred_proba, fn_cost=10.0, fp_cost=1.0):
        """Trouve le seuil optimal basé sur le coût métier"""
        thresholds = np.linspace(0, 1, 100)
        costs = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred_thresh)
            tn, fp, fn, tp = cm.ravel()
            cost = (fn * fn_cost) + (fp * fp_cost)
            costs.append(cost)
        
        optimal_idx = np.argmin(costs)
        return thresholds[optimal_idx], costs[optimal_idx]
    
    def _create_plots(self, y_true, y_pred, y_pred_proba, model_name):
        """Crée les visualisations pour MLFlow"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0])
        axes[0,0].set_title(f'Confusion Matrix - {model_name}')
        
        # Distribution des probabilités
        axes[0,1].hist(y_pred_proba[y_true==0], bins=30, alpha=0.7, label='No Default', color='green')
        axes[0,1].hist(y_pred_proba[y_true==1], bins=30, alpha=0.7, label='Default', color='red')
        axes[0,1].set_title('Probability Distribution')
        axes[0,1].legend()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        axes[1,0].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_pred_proba):.3f}')
        axes[1,0].plot([0, 1], [0, 1], 'k--')
        axes[1,0].set_title('ROC Curve')
        axes[1,0].legend()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        axes[1,1].plot(recall, precision)
        axes[1,1].set_title('Precision-Recall Curve')
        
        plt.tight_layout()
        mlflow.log_figure(fig, f"{model_name}_plots.png")
        plt.show()


    def _create_threshold_optimization_plots(self, y_true, y_pred_proba, model_name):
        """
        Create plots for threshold optimization and confusion matrix comparison.
    
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Model name for logging
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
        # 1. Business Cost vs Threshold Curve
        thresholds = np.linspace(0, 1, 100)
        costs = []
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            cost = calculate_business_cost(y_true, y_pred_thresh)
            costs.append(cost)
    
        optimal_threshold, optimal_cost = self._find_optimal_threshold(y_true, y_pred_proba)
        default_threshold = 0.5
        default_cost = calculate_business_cost(
            y_true, 
            (y_pred_proba >= default_threshold).astype(int)
        )
    
        axes[0, 0].plot(thresholds, costs, 'b-', linewidth=2, label='Business Cost')
        axes[0, 0].axvline(x=optimal_threshold, color='green', linestyle='--', 
                       linewidth=2, label=f'Optimal: {optimal_threshold:.3f} (Cost: {optimal_cost:.0f})')
        axes[0, 0].axvline(x=default_threshold, color='red', linestyle='--', 
                       linewidth=2, label=f'Default: {default_threshold:.3f} (Cost: {default_cost:.0f})')
        axes[0, 0].axhline(y=optimal_cost, color='green', linestyle=':', alpha=0.5)
        axes[0, 0].axhline(y=default_cost, color='red', linestyle=':', alpha=0.5)
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Business Cost (10×FN + 1×FP)')
        axes[0, 0].set_title('Business Cost vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
        # 2. Confusion Matrix - Default Threshold (0.5)
        y_pred_default = (y_pred_proba >= default_threshold).astype(int)
        cm_default = confusion_matrix(y_true, y_pred_default)
        tn_default, fp_default, fn_default, tp_default = cm_default.ravel()
    
        sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
        axes[0, 1].set_title(f'Confusion Matrix - Default Threshold (0.5)\n'
                         f'Cost: {default_cost:.0f} | FN: {fn_default}, FP: {fp_default}')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')
    
        # 3. Confusion Matrix - Optimal Threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        cm_optimal = confusion_matrix(y_true, y_pred_optimal)
        tn_optimal, fp_optimal, fn_optimal, tp_optimal = cm_optimal.ravel()
    
        sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0],
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
        axes[1, 0].set_title(f'Confusion Matrix - Optimal Threshold ({optimal_threshold:.3f})\n'
                         f'Cost: {optimal_cost:.0f} | FN: {fn_optimal}, FP: {fp_optimal}')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
    
        # 4. Comparison Summary Table
        axes[1, 1].axis('off')
    
        comparison_data = {
            'Metric': [
                'Threshold',
                'Business Cost',
                'False Negatives (FN)',
                'False Positives (FP)',
                'True Positives (TP)',
                'True Negatives (TN)',
                'Recall (Sensitivity)',
                'Precision',
                'Cost Reduction'
            ],
            'Default (0.5)': [
                f'{default_threshold:.3f}',
                f'{default_cost:.0f}',
                f'{fn_default}',
                f'{fp_default}',
                f'{tp_default}',
                f'{tn_default}',
                f'{tp_default/(tp_default+fn_default):.3f}' if (tp_default+fn_default) > 0 else '0.000',
                f'{tp_default/(tp_default+fp_default):.3f}' if (tp_default+fp_default) > 0 else '0.000',
                '-'
            ],
            'Optimal': [
                f'{optimal_threshold:.3f}',
                f'{optimal_cost:.0f}',
                f'{fn_optimal}',
                f'{fp_optimal}',
                f'{tp_optimal}',
                f'{tn_optimal}',
                f'{tp_optimal/(tp_optimal+fn_optimal):.3f}' if (tp_optimal+fn_optimal) > 0 else '0.000',
                f'{tp_optimal/(tp_optimal+fp_optimal):.3f}' if (tp_optimal+fp_optimal) > 0 else '0.000',
                f'{((default_cost - optimal_cost) / default_cost * 100):.1f}%'
            ]
        }
    
        comparison_df = pd.DataFrame(comparison_data)
        table = axes[1, 1].table(cellText=comparison_df.values,
                              colLabels=comparison_df.columns,
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
    
        # Style the header
        for i in range(len(comparison_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
        # Highlight improvements
        for i in range(1, len(comparison_df)):
            if comparison_df.iloc[i, 0] in ['Business Cost', 'False Negatives (FN)', 'Cost Reduction']:
                table[(i, 2)].set_facecolor('#C8E6C9')  # Light green for optimal
    
        axes[1, 1].set_title('Threshold Comparison Summary', pad=20, fontsize=12, weight='bold')
    
        plt.tight_layout()
        mlflow.log_figure(fig, f"{model_name}_threshold_optimization.png")
        plt.close(fig)
    
    def compare_models(self, models_config, X_train, X_test, y_train, y_test):
        """Compare plusieurs modèles"""
        results = []
        
        for model_config in models_config:
            metrics = self.log_experiment(
                model=model_config["model"],
                model_name=model_config["name"],
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                params=model_config.get("params", {}),
                tags=model_config.get("tags", {})
            )
            results.append({
                "model": model_config["name"],
                "metrics": metrics
            })
        
        # Afficher la comparaison
        print("\n📊 Comparaison des modèles:")
        print("-" * 80)
        print(f"{'Modèle':<20} {'Accuracy':<10} {'AUC':<10} {'Business Cost':<15}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['model']:<20} {result['metrics']['accuracy']:<10.3f} {result['metrics']['auc_roc']:<10.3f} {result['metrics']['business_cost']:<15.1f}")
        
        # Meilleur modèle
        best_model = min(results, key=lambda x: x['metrics']['business_cost'])
        print(f"\n🏆 Meilleur modèle: {best_model['model']} (Coût métier: {best_model['metrics']['business_cost']:.1f})")
        
        return results
    
    def get_best_model(self, metric="business_cost", ascending=True):
        """Récupère le meilleur modèle selon une métrique"""
        try:
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
            )
            
            if runs:
                best_run = runs[0]
                return {
                    "run_id": best_run.info.run_id,
                    "metric_value": best_run.data.metrics.get(metric),
                    "model_uri": f"runs:/{best_run.info.run_id}/model"
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Erreur lors de la récupération du meilleur modèle: {e}")
            return None
    
    def show_ui_info(self):
        """Affiche les informations pour accéder à l'interface MLFlow"""
        project_root = Path(__file__).resolve().parent.parent.parent
        backend_uri = f"sqlite:///{project_root}/mlruns/mlflow.db"
        print("🌐 Interface MLFlow disponible sur: http://localhost:5000")
        print("📊 Pour démarrer l'interface (avec le bon backend SQLite), exécutez dans le terminal:")
        print(f"   source .venv/bin/activate && mlflow ui --backend-store-uri \"{backend_uri}\" --host 127.0.0.1 --port 5000")
        print("\n📈 Vous pouvez maintenant:")
        print("   - Voir toutes vos expérimentations")
        print("   - Comparer les métriques")
        print("   - Visualiser les graphiques")
        print("   - Télécharger les modèles")
        print("   - Gérer le registry des modèles")
    
    def cross_validate_model(self, model, X, y, cv=5, model_name="Model", 
                            params=None, tags=None, log_to_mlflow=True,
                            create_plots=True):
        """
        Run K-Fold cross-validation and log aggregated metrics to MLflow.
        
        Args:
            model: Scikit-learn compatible model (unfitted)
            X: Feature matrix
            y: Target vector
            cv: Number of folds (default 5)
            model_name: Name for logging
            params: Model parameters to log
            tags: Additional tags for MLflow
            log_to_mlflow: Whether to log to MLflow
            create_plots: Whether to create and log plots
        
        Returns:
            Dictionary with per-fold and aggregated metrics
        """
        from sklearn.base import clone
        
        # Initialize StratifiedKFold for class balance preservation
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Storage for fold results
        fold_metrics = {
            'auc': [], 'accuracy': [], 'precision': [], 
            'recall': [], 'f1': [], 'business_cost': []
        }
        all_y_true = []
        all_y_pred_proba = []
        
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION: {model_name} ({cv} folds)")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            fold_start = time.time()
            
            # Split data
            if isinstance(X, pd.DataFrame):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
            else:
                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
            
            if isinstance(y, pd.Series):
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
            else:
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]
            
            # Clone and train model
            model_clone = clone(model)
            model_clone.fit(X_train_fold, y_train_fold)
            
            # Predictions (only probabilities - no class predictions yet)
            y_pred_proba = model_clone.predict_proba(X_val_fold)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)  # Default threshold for metrics
            
            # Store for aggregated plots
            all_y_true.extend(y_val_fold)
            all_y_pred_proba.extend(y_pred_proba)
            
            # Calculate fold metrics
            fold_auc = roc_auc_score(y_val_fold, y_pred_proba)
            fold_acc = accuracy_score(y_val_fold, y_pred)
            fold_prec = precision_score(y_val_fold, y_pred, zero_division=0)
            fold_rec = recall_score(y_val_fold, y_pred, zero_division=0)
            fold_f1 = f1_score(y_val_fold, y_pred, zero_division=0)
            
            # Business cost
            cm = confusion_matrix(y_val_fold, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fold_cost = (fn * FN_COST) + (fp * FP_COST)
            
            # Store metrics
            fold_metrics['auc'].append(fold_auc)
            fold_metrics['accuracy'].append(fold_acc)
            fold_metrics['precision'].append(fold_prec)
            fold_metrics['recall'].append(fold_rec)
            fold_metrics['f1'].append(fold_f1)
            fold_metrics['business_cost'].append(fold_cost)
            
            fold_time = time.time() - fold_start
            print(f"  Fold {fold_idx + 1}/{cv}: AUC={fold_auc:.4f}, Acc={fold_acc:.4f}, Cost={fold_cost:.0f} ({fold_time:.1f}s)")
        
        total_time = time.time() - start_time
        
        # Calculate aggregated metrics
        results = {
            'model_name': model_name,
            'n_folds': cv,
            'fold_metrics': fold_metrics,
            'cv_auc_mean': np.mean(fold_metrics['auc']),
            'cv_auc_std': np.std(fold_metrics['auc']),
            'cv_accuracy_mean': np.mean(fold_metrics['accuracy']),
            'cv_accuracy_std': np.std(fold_metrics['accuracy']),
            'cv_precision_mean': np.mean(fold_metrics['precision']),
            'cv_precision_std': np.std(fold_metrics['precision']),
            'cv_recall_mean': np.mean(fold_metrics['recall']),
            'cv_recall_std': np.std(fold_metrics['recall']),
            'cv_f1_mean': np.mean(fold_metrics['f1']),
            'cv_f1_std': np.std(fold_metrics['f1']),
            'cv_business_cost_mean': np.mean(fold_metrics['business_cost']),
            'cv_business_cost_std': np.std(fold_metrics['business_cost']),
            'training_time': total_time
        }
        
        # Find optimal threshold on aggregated OOF predictions
        all_y_true = np.array(all_y_true)
        all_y_pred_proba = np.array(all_y_pred_proba)
        optimal_threshold, optimal_cost = self._find_optimal_threshold(all_y_true, all_y_pred_proba)
        results['optimal_threshold'] = optimal_threshold
        results['optimal_business_cost'] = optimal_cost
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {model_name}")
        print(f"{'='*60}")
        print(f"  AUC:           {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
        print(f"  Accuracy:      {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
        print(f"  Business Cost: {results['cv_business_cost_mean']:.1f} ± {results['cv_business_cost_std']:.1f}")
        print(f"  Optimal Threshold: {optimal_threshold:.3f} (Cost: {optimal_cost:.1f})")
        print(f"  Training Time: {total_time:.1f}s")
        
        # Log to MLflow
        if log_to_mlflow:
            run_tags = {
                "model_type": model_name,
                "domain": "credit-scoring",
                "cv_folds": str(cv),
                "evaluation_type": "cross_validation"
            }
            if tags:
                run_tags.update(tags)
            
            with mlflow.start_run(run_name=f"{model_name}_CV{cv}", tags=run_tags):
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                mlflow.log_params({
                    "cv_folds": cv,
                    "n_samples": len(y),
                    "n_features": X.shape[1],
                    "positive_rate": float(y.mean()),
                    "class_imbalance_ratio": float((1 - y.mean()) / y.mean())
                })
                
                # Log metrics
                mlflow.log_metrics({
                    "cv_auc_mean": results['cv_auc_mean'],
                    "cv_auc_std": results['cv_auc_std'],
                    "cv_accuracy_mean": results['cv_accuracy_mean'],
                    "cv_accuracy_std": results['cv_accuracy_std'],
                    "cv_precision_mean": results['cv_precision_mean'],
                    "cv_recall_mean": results['cv_recall_mean'],
                    "cv_f1_mean": results['cv_f1_mean'],
                    "cv_business_cost_mean": results['cv_business_cost_mean'],
                    "cv_business_cost_std": results['cv_business_cost_std'],
                    "optimal_threshold": optimal_threshold,
                    "optimal_business_cost": optimal_cost,
                    "training_time_seconds": total_time
                })
                
                # Create and log plots
                if create_plots:
                    self._create_cv_plots(all_y_true, all_y_pred_proba, fold_metrics, model_name)
                    self._create_threshold_optimization_plots(all_y_true, all_y_pred_proba, model_name)
                
                print(f"\n✅ Logged to MLflow: {model_name}")
        
        return results
    
    def _create_cv_plots(self, y_true, y_pred_proba, fold_metrics, model_name):
        """Create cross-validation specific plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. ROC Curve (aggregated OOF)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title(f'ROC Curve (OOF) - {model_name}')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Probability Distribution
        axes[0, 1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.6, 
                       label='No Default (0)', color='green', density=True)
        axes[0, 1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.6, 
                       label='Default (1)', color='red', density=True)
        axes[0, 1].set_xlabel('Predicted Probability')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Probability Distribution by Class')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. AUC per Fold
        folds = range(1, len(fold_metrics['auc']) + 1)
        axes[1, 0].bar(folds, fold_metrics['auc'], color='steelblue', alpha=0.7)
        axes[1, 0].axhline(y=np.mean(fold_metrics['auc']), color='red', 
                          linestyle='--', linewidth=2, label=f"Mean: {np.mean(fold_metrics['auc']):.4f}")
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('AUC by Fold')
        axes[1, 0].legend()
        axes[1, 0].set_xticks(folds)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Business Cost per Fold
        axes[1, 1].bar(folds, fold_metrics['business_cost'], color='coral', alpha=0.7)
        axes[1, 1].axhline(y=np.mean(fold_metrics['business_cost']), color='red',
                          linestyle='--', linewidth=2, label=f"Mean: {np.mean(fold_metrics['business_cost']):.1f}")
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('Business Cost')
        axes[1, 1].set_title('Business Cost by Fold (10×FN + 1×FP)')
        axes[1, 1].legend()
        axes[1, 1].set_xticks(folds)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        mlflow.log_figure(fig, f"{model_name}_cv_plots.png")
        plt.close(fig)
    
    def log_optuna_trial(self, trial_number, params, metrics, model_name="Model"):
        """
        Log a single Optuna trial to MLflow.
        
        Args:
            trial_number: Trial number from Optuna
            params: Hyperparameters tried in this trial
            metrics: Metrics from cross-validation
            model_name: Name for logging
        """
        with mlflow.start_run(run_name=f"{model_name}_optuna_trial_{trial_number:03d}",
                             tags={"model_type": model_name, "optimization": "optuna"}):
            mlflow.log_params(params)
            mlflow.log_params({"trial_number": trial_number})
            
            # Log the key metrics
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
            else:
                mlflow.log_metric("cv_auc_mean", metrics)
    
    def log_best_model(self, model, model_name, X_train, y_train, 
                       best_params, cv_results, tags=None):
        """
        Train and log the final best model to MLflow registry.
        
        Args:
            model: Fitted model with best hyperparameters
            model_name: Name for the model
            X_train: Training features
            y_train: Training target
            best_params: Best hyperparameters found
            cv_results: Cross-validation results
            tags: Additional tags
        """
        run_tags = {
            "model_type": model_name,
            "domain": "credit-scoring",
            "optimization": "optuna",
            "best_model": "true"
        }
        if tags:
            run_tags.update(tags)
        
        with mlflow.start_run(run_name=f"{model_name}_best", tags=run_tags):
            # Log parameters
            mlflow.log_params(best_params)
            mlflow.log_params({
                "n_samples": len(y_train),
                "n_features": X_train.shape[1]
            })
            
            # Log CV metrics
            if isinstance(cv_results, dict):
                for key, value in cv_results.items():
                    if isinstance(value, (int, float)) and not key.startswith('fold_'):
                        mlflow.log_metric(key, value)
            
            # Log model - Detect model type and use appropriate MLflow flavor
            from lightgbm import LGBMClassifier
            from sklearn.ensemble import RandomForestClassifier
            from xgboost import XGBClassifier
            
            # Infer model signature
            signature = mlflow.models.infer_signature(
                X_train, 
                model.predict_proba(X_train)[:, 1]
            )
            
            # Use appropriate MLflow flavor based on model type
            if isinstance(model, LGBMClassifier):
                mlflow.lightgbm.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=f"{model_name}_best"
                )
            elif isinstance(model, XGBClassifier):
                mlflow.xgboost.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=f"{model_name}_best"
                )
            elif isinstance(model, RandomForestClassifier):
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=f"{model_name}_best"
                )
            else:
                # Fallback for other sklearn models
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=f"{model_name}_best"
                )
            
            run_id = mlflow.active_run().info.run_id
            print(f"\n✅ Best model logged to MLflow")
            print(f"   Run ID: {run_id}")
            print(f"   Model URI: runs:/{run_id}/model")
            
            return run_id


def create_optuna_objective(model_class, X, y, param_space, cv=5, 
                           fn_cost=10.0, fp_cost=1.0):
    """
    Create an Optuna objective function for hyperparameter tuning.
    
    Args:
        model_class: Scikit-learn model class (not instance)
        X: Feature matrix
        y: Target vector
        param_space: Function that takes trial and returns params dict
        cv: Number of cross-validation folds
        fn_cost: False Negative cost
        fp_cost: False Positive cost
    
    Returns:
        Objective function for Optuna
    """
    from sklearn.base import clone
    
    def objective(trial):
        # Get hyperparameters from trial
        params = param_space(trial)
        
        # Create model with these params
        model = model_class(**params)
        
        # Perform cross-validation
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        auc_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            if isinstance(X, pd.DataFrame):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
            else:
                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
            
            if isinstance(y, pd.Series):
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
            else:
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]
            
            model_clone = clone(model)
            model_clone.fit(X_train_fold, y_train_fold)
            y_pred_proba = model_clone.predict_proba(X_val_fold)[:, 1]
            auc_scores.append(roc_auc_score(y_val_fold, y_pred_proba))
        
        return np.mean(auc_scores)
    
    return objective


# Fonction de convenance pour utilisation rapide
def quick_mlflow_setup(experiment_name="credit-scoring-notebook"):
    """Configuration rapide de MLFlow pour les notebooks"""
    return NotebookMLFlow(experiment_name=experiment_name)


def calculate_business_cost(y_true, y_pred, fn_cost=FN_COST, fp_cost=FP_COST):
    """
    Calculate business cost: FN costs 10x more than FP.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        fn_cost: Cost of false negatives (default 10)
        fp_cost: Cost of false positives (default 1)
    
    Returns:
        Total business cost
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return (fn * fn_cost) + (fp * fp_cost)


def find_optimal_threshold(y_true, y_pred_proba, fn_cost=FN_COST, fp_cost=FP_COST):
    """
    Find the optimal classification threshold based on business cost.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        fn_cost: Cost of false negatives (default 10)
        fp_cost: Cost of false positives (default 1)
    
    Returns:
        Tuple of (optimal_threshold, minimum_cost)
    """
    thresholds = np.linspace(0, 1, 100)
    costs = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        cost = calculate_business_cost(y_true, y_pred, fn_cost, fp_cost)
        costs.append(cost)
    
    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx], costs[optimal_idx]



