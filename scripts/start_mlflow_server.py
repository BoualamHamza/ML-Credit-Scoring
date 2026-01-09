#!/usr/bin/env python3
"""
Script pour démarrer le serveur MLFlow
"""
import subprocess
import sys
from pathlib import Path
from config.mlflow_config import MLFLOW_SERVER_CONFIG

def start_mlflow_server():
    """Démarre le serveur MLFlow avec la configuration appropriée"""
    
    # Vérifier que MLFlow est installé
    try:
        import mlflow
        print("✅ MLFlow est installé")
    except ImportError:
        print("❌ MLFlow n'est pas installé. Installez-le avec: pip install mlflow")
        sys.exit(1)
    
    # Construire la commande MLFlow
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", MLFLOW_SERVER_CONFIG["backend_store_uri"],
        "--default-artifact-root", MLFLOW_SERVER_CONFIG["default_artifact_root"],
        "--host", MLFLOW_SERVER_CONFIG["host"],
        "--port", str(MLFLOW_SERVER_CONFIG["port"]),
        "--workers", str(MLFLOW_SERVER_CONFIG["workers"])
    ]
    
    print("🚀 Démarrage du serveur MLFlow...")
    print(f"📍 URI de tracking: {MLFLOW_SERVER_CONFIG['backend_store_uri']}")
    print(f"📁 Artifacts root: {MLFLOW_SERVER_CONFIG['default_artifact_root']}")
    print(f"🌐 Interface web: http://{MLFLOW_SERVER_CONFIG['host']}:{MLFLOW_SERVER_CONFIG['port']}")
    print("\n" + "="*50)
    
    try:
        # Démarrer le serveur
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du serveur MLFlow")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du démarrage du serveur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_mlflow_server()
