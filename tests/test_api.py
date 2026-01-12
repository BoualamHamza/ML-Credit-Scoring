"""
Unit tests for Credit Scoring API

Tests cover:
- API health and model info endpoints
- Prediction by client ID
- Error handling
"""
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
import pickle

# Add parent directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.api.api import app, load_model_from_local

# Create test client
client = TestClient(app)


@pytest.fixture(scope="module")
def test_client():
    """Create test client for API"""
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_client_id():
    """Get a sample client ID from the dataset if available"""
    # Try sample dataset first (for Azure deployment)
    sample_path = BASE_DIR / "src" / "dataset" / "features_train_sample_1000.pkl"
    features_path = BASE_DIR / "src" / "dataset" / "features_train.pkl"
    
    # Try sample dataset first
    if sample_path.exists():
        with open(sample_path, 'rb') as f:
            df = pickle.load(f)
    elif features_path.exists():
        with open(features_path, 'rb') as f:
            df = pickle.load(f)
    else:
        return 100001  # Default test ID
    
    # Extract client ID
    if 'SK_ID_CURR' in df.columns:
        return int(df['SK_ID_CURR'].iloc[0])
    # If SK_ID_CURR is the index
    if hasattr(df.index, 'name') and df.index.name == 'SK_ID_CURR':
        return int(df.index[0])
    # Try to get from index if it's already set
    if not isinstance(df.index, pd.RangeIndex):
        return int(df.index[0])
    
    return 100001  # Default test ID


def test_health_check(test_client):
    """Test health check endpoint"""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    # Model may or may not be loaded depending on test environment
    assert isinstance(data["model_loaded"], bool)


def test_model_info(test_client):
    """Test model info endpoint"""
    response = test_client.get("/model/info")
    
    # If model is loaded, should return 200
    if response.status_code == 200:
        data = response.json()
        assert "model_type" in data
        assert "n_features" in data
        assert "optimal_threshold" in data
        assert "feature_names" in data
        assert "model_loaded" in data
        assert data["optimal_threshold"] == 0.475
    else:
        # If model not loaded, should return 503
        assert response.status_code == 503


def test_predict_client_id_valid(test_client, sample_client_id):
    """Test prediction with valid client ID"""
    response = test_client.post(
        "/predict/client_id",
        json={"client_id": sample_client_id}
    )
    
    if response.status_code == 200:
        data = response.json()
        assert "client_id" in data
        assert "probability" in data
        assert "prediction" in data
        assert "threshold" in data
        assert "recommendation" in data
        
        # Validate data types and ranges
        assert isinstance(data["client_id"], int)
        assert isinstance(data["probability"], float)
        assert 0.0 <= data["probability"] <= 1.0
        assert data["prediction"] in [0, 1]
        assert data["threshold"] == 0.475
        assert isinstance(data["recommendation"], str)
    elif response.status_code == 404:
        # Client ID not found - acceptable if dataset not loaded
        assert "not found" in response.json()["detail"].lower()
    elif response.status_code == 503:
        # Model not loaded - acceptable in test environment
        pass
    else:
        pytest.fail(f"Unexpected status code: {response.status_code}")


def test_predict_client_id_invalid(test_client):
    """Test prediction with invalid client ID"""
    invalid_id = 999999999
    response = test_client.post(
        "/predict/client_id",
        json={"client_id": invalid_id}
    )
    
    # Should return 404 if dataset is loaded, or 503 if model not loaded
    assert response.status_code in [404, 503]
    
    if response.status_code == 404:
        assert "not found" in response.json()["detail"].lower()


def test_predict_client_id_negative(test_client):
    """Test prediction with negative client ID"""
    response = test_client.post(
        "/predict/client_id",
        json={"client_id": -1}
    )
    
    # Should return 404 or 503
    assert response.status_code in [404, 503, 422]  # 422 for validation error


def test_optimal_threshold():
    """Test that optimal threshold is correctly set"""
    from src.api.api import optimal_threshold
    assert optimal_threshold == 0.475


def test_prediction_probability_range(test_client, sample_client_id):
    """Test that prediction probabilities are in valid range [0, 1]"""
    response = test_client.post(
        "/predict/client_id",
        json={"client_id": sample_client_id}
    )
    
    if response.status_code == 200:
        data = response.json()
        probability = data["probability"]
        assert 0.0 <= probability <= 1.0, f"Probability {probability} is not in [0, 1]"


def test_prediction_binary(test_client, sample_client_id):
    """Test that predictions are binary (0 or 1)"""
    response = test_client.post(
        "/predict/client_id",
        json={"client_id": sample_client_id}
    )
    
    if response.status_code == 200:
        data = response.json()
        prediction = data["prediction"]
        assert prediction in [0, 1], f"Prediction {prediction} is not binary"


def test_recommendation_format(test_client, sample_client_id):
    """Test that recommendation is a non-empty string"""
    response = test_client.post(
        "/predict/client_id",
        json={"client_id": sample_client_id}
    )
    
    if response.status_code == 200:
        data = response.json()
        recommendation = data["recommendation"]
        assert isinstance(recommendation, str)
        assert len(recommendation) > 0


def test_error_handling_invalid_json(test_client):
    """Test error handling for invalid JSON"""
    response = test_client.post(
        "/predict/client_id",
        json={"invalid": "data"}
    )
    
    # Should return 422 (validation error) or 503
    assert response.status_code in [422, 503]


def test_error_handling_missing_field(test_client):
    """Test error handling for missing required field"""
    response = test_client.post(
        "/predict/client_id",
        json={}
    )
    
    # Should return 422 (validation error)
    assert response.status_code == 422


def test_get_client_ids(test_client):
    """Test getting list of available client IDs"""
    response = test_client.get("/clients/ids")
    
    if response.status_code == 200:
        data = response.json()
        assert "client_ids" in data
        assert "total" in data
        assert isinstance(data["client_ids"], list)
        assert isinstance(data["total"], int)
        assert data["total"] == len(data["client_ids"])
        # If we have client IDs, they should be integers
        if len(data["client_ids"]) > 0:
            assert isinstance(data["client_ids"][0], int)
    elif response.status_code == 503:
        # Model not loaded - acceptable in test environment
        pass
    else:
        pytest.fail(f"Unexpected status code: {response.status_code}")


def test_predict_client_id_shap(test_client, sample_client_id):
    """Test SHAP values endpoint"""
    response = test_client.post(
        "/predict/client_id/shap",
        json={"client_id": sample_client_id}
    )
    
    if response.status_code == 200:
        data = response.json()
        assert "client_id" in data
        assert "shap_values" in data
        assert "top_features" in data
        
        assert data["client_id"] == sample_client_id
        assert isinstance(data["shap_values"], list)
        assert isinstance(data["top_features"], int)
        assert data["top_features"] > 0
        
        # Validate SHAP value structure
        if len(data["shap_values"]) > 0:
            shap_item = data["shap_values"][0]
            assert "feature" in shap_item
            assert "value" in shap_item
            assert "importance" in shap_item
            assert isinstance(shap_item["feature"], str)
            assert isinstance(shap_item["value"], float)
            assert isinstance(shap_item["importance"], float)
            assert shap_item["importance"] >= 0  # Importance is absolute value
    elif response.status_code == 404:
        # Client ID not found - acceptable if dataset not loaded
        assert "not found" in response.json()["detail"].lower()
    elif response.status_code == 503:
        # Model not loaded - acceptable in test environment
        pass
    else:
        pytest.fail(f"Unexpected status code: {response.status_code}")


def test_predict_client_id_shap_with_top_n(test_client, sample_client_id):
    """Test SHAP values endpoint with custom top_n parameter"""
    response = test_client.post(
        "/predict/client_id/shap?top_n=10",
        json={"client_id": sample_client_id}
    )
    
    if response.status_code == 200:
        data = response.json()
        assert "top_features" in data
        assert data["top_features"] <= 10
        assert len(data["shap_values"]) <= 10
    elif response.status_code in [404, 503]:
        # Acceptable if dataset/model not loaded
        pass


def test_predict_client_id_shap_invalid_id(test_client):
    """Test SHAP values endpoint with invalid client ID"""
    invalid_id = 999999999
    response = test_client.post(
        "/predict/client_id/shap",
        json={"client_id": invalid_id}
    )
    
    # Should return 404 if dataset is loaded, or 503 if model not loaded
    assert response.status_code in [404, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
