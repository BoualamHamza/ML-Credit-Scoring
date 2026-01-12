#!/bin/bash

# Startup script for Azure Web App
# This script is executed when the Azure Web App starts

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/home/site/wwwroot"

# Change to the application directory (Azure uses /home/site/wwwroot)
cd /home/site/wwwroot || cd ${HOME}/site/wwwroot || pwd

echo "📂 Current directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "📦 Python path: $(which python)"

# Install/upgrade pip and install dependencies
echo "📦 Installing dependencies..."
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Check if sample dataset exists, create it if not
SAMPLE_DATASET="src/dataset/features_train_sample_1000.pkl"
FULL_DATASET="src/dataset/features_train.pkl"

if [ ! -f "$SAMPLE_DATASET" ]; then
    echo "📦 Sample dataset not found. Checking if full dataset exists..."
    if [ -f "$FULL_DATASET" ]; then
        echo "✅ Full dataset found. Creating sample dataset..."
        python scripts/create_sample_dataset.py
        if [ $? -eq 0 ]; then
            echo "✅ Sample dataset created successfully"
        else
            echo "⚠️  Warning: Failed to create sample dataset. API will use fallback."
        fi
    else
        echo "⚠️  Warning: Neither sample nor full dataset found. API may not work correctly."
    fi
else
    echo "✅ Sample dataset already exists"
fi

# Run the FastAPI application
# Azure Web App uses PORT environment variable
# Default to 8000 if not set
PORT=${PORT:-8000}

echo "🚀 Starting FastAPI server on port ${PORT}..."

# Start uvicorn server
python -m uvicorn src.api.api:app --host 0.0.0.0 --port ${PORT}
