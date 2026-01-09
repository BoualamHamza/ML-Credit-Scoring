#!/bin/bash
# Script pour démarrer MLFlow UI

echo "🚀 Démarrage de l'interface MLFlow..."
echo "📍 URI de tracking: sqlite:////Users/hamzaboualam/Downloads/OpenClassRoom- projects/P7/mlruns/mlflow.db"
echo "🌐 Interface web: http://localhost:5001"
echo ""

# Activer l'environnement virtuel
source .venv/bin/activate

# Démarrer MLFlow UI
mlflow ui --backend-store-uri "sqlite:////Users/hamzaboualam/Downloads/OpenClassRoom- projects/P7/mlruns/mlflow.db" --host 127.0.0.1 --port 5001
