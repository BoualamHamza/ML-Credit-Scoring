#!/bin/bash

# Startup script for Azure Web App
# This script is executed when the Azure Web App starts

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${HOME}/site/wwwroot"

# Run the FastAPI application
# Azure Web App uses PORT environment variable
# Default to 8000 if not set
PORT=${PORT:-8000}

# Start uvicorn server
# Note: Azure Web App runs from the wwwroot directory
cd ${HOME}/site/wwwroot || cd /home/site/wwwroot || pwd
python -m uvicorn src.api.api:app --host 0.0.0.0 --port ${PORT}
