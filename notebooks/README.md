# Guide d'utilisation MLFlow dans les Notebooks

## 🚀 Démarrage Rapide

### 1. Configuration dans votre notebook

```python
# Import de l'utilitaire
from src.utils.notebook_mlflow import quick_mlflow_setup

# Initialiser MLFlow
mlflow_tracker = quick_mlflow_setup("mon-experience")
```

### 2. Entraîner et logger un modèle

```python
from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner et logger automatiquement
metrics = mlflow_tracker.log_experiment(
    model=model,
    model_name="RandomForest",
    X_train=X_train, X_test=X_test, 
    y_train=y_train, y_test=y_test,
    params={"n_estimators": 100, "random_state": 42},
    tags={"baseline": "true"}
)
```

### 3. Comparer plusieurs modèles

```python
models_config = [
    {
        "model": LogisticRegression(random_state=42),
        "name": "LogisticRegression",
        "params": {"random_state": 42}
    },
    {
        "model": GradientBoostingClassifier(n_estimators=100),
        "name": "GradientBoosting", 
        "params": {"n_estimators": 100}
    }
]

# Comparer tous les modèles
results = mlflow_tracker.compare_models(
    models_config, X_train, X_test, y_train, y_test
)
```

## 📊 Fonctionnalités Automatiques

### Métriques Trackées
- ✅ **Accuracy**: Précision globale
- ✅ **Precision**: Précision
- ✅ **Recall**: Rappel  
- ✅ **F1-Score**: Score F1
- ✅ **AUC-ROC**: Capacité de discrimination
- ✅ **Business Cost**: Coût métier (FN = 10x FP)
- ✅ **Optimal Threshold**: Seuil optimisé

### Visualisations Automatiques
- 📈 Matrice de confusion
- 📊 Distribution des probabilités
- 📈 Courbe ROC
- 📈 Courbe Precision-Recall

### Informations sur les Données
- 📊 Nombre d'échantillons train/test
- 🎯 Taux de défaut
- ⚖️ Ratio de déséquilibre des classes
- 🔢 Nombre de features

## 🎯 Exemples d'Utilisation

### Exemple 1: Modèle Simple
```python
# Configuration
mlflow_tracker = quick_mlflow_setup("credit-scoring")

# Modèle
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement et logging
metrics = mlflow_tracker.log_experiment(
    model=model,
    model_name="RandomForest",
    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)
```

### Exemple 2: Comparaison de Modèles
```python
# Liste des modèles à comparer
models_config = [
    {
        "model": LogisticRegression(random_state=42),
        "name": "LogisticRegression"
    },
    {
        "model": RandomForestClassifier(n_estimators=100, random_state=42),
        "name": "RandomForest"
    },
    {
        "model": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "name": "GradientBoosting"
    }
]

# Comparaison automatique
results = mlflow_tracker.compare_models(
    models_config, X_train, X_test, y_train, y_test
)
```

### Exemple 3: Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Configuration de la grille
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='roc_auc'
)

# Entraînement
grid_search.fit(X_train, y_train)

# Logging du meilleur modèle
metrics = mlflow_tracker.log_experiment(
    model=grid_search.best_estimator_,
    model_name="RandomForest_GridSearch",
    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    params=grid_search.best_params_,
    tags={"hyperparameter_tuning": "true"}
)
```

## 🌐 Interface MLFlow

### Accès à l'Interface
1. **Démarrer le serveur MLFlow**:
   ```bash
   source .venv/bin/activate && mlflow ui
   ```

2. **Ouvrir dans le navigateur**: http://localhost:5000

### Fonctionnalités de l'Interface
- 📊 **Expérimentations**: Voir toutes vos expérimentations
- 🔍 **Runs**: Détails de chaque run avec métriques
- 📈 **Comparaison**: Comparer les modèles côte à côte
- 📦 **Modèles**: Registry des modèles entraînés
- 📊 **Graphiques**: Visualiser les courbes et matrices

## 🎯 Bonnes Pratiques

### 1. Nommage des Expérimentations
```python
# Utilisez des noms descriptifs
mlflow_tracker = quick_mlflow_setup("credit-scoring-feature-engineering")
mlflow_tracker = quick_mlflow_setup("credit-scoring-hyperparameter-tuning")
```

### 2. Tags Informatifs
```python
tags = {
    "baseline": "true",
    "feature_engineering": "none",
    "data_version": "v1.0",
    "business_context": "credit-scoring"
}
```

### 3. Paramètres Détaillés
```python
params = {
    "model_type": "RandomForest",
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "preprocessing": "StandardScaler",
    "feature_selection": "None"
}
```

## 🔧 Configuration Avancée

### URI de Tracking Personnalisée
```python
from src.utils.notebook_mlflow import NotebookMLFlow

# Configuration personnalisée
mlflow_tracker = NotebookMLFlow(
    experiment_name="mon-experience",
    tracking_uri="sqlite:///custom/path/mlflow.db"
)
```

### Récupération du Meilleur Modèle
```python
# Récupérer le meilleur modèle selon le coût métier
best_model = mlflow_tracker.get_best_model("business_cost", ascending=True)
print(f"Meilleur modèle: {best_model['run_id']}")
print(f"Coût métier: {best_model['metric_value']}")
```

## 🆘 Dépannage

### Problèmes Courants
1. **Import Error**: Vérifiez que vous êtes dans le bon répertoire
2. **MLFlow URI**: Assurez-vous que le chemin vers la base de données est correct
3. **Permissions**: Vérifiez les permissions sur les dossiers `mlruns/` et `mlartifacts/`

### Logs et Debug
```python
# Activer les logs MLFlow
import logging
logging.basicConfig(level=logging.INFO)
```

## 📚 Ressources

- [Documentation MLFlow](https://mlflow.org/docs/latest/index.html)
- [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLFlow Models](https://mlflow.org/docs/latest/models.html)



