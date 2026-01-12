"""
FastAPI application for Credit Scoring Model Serving

This API provides endpoints to:
- Get model information
- Make predictions by client ID (SK_ID_CURR)
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import sys
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
import logging
import shap

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import project utilities
from src.utils.feature_io import load_feature_names

# Global variables for model and data
model = None
feature_names = None
full_dataset = None
client_ids = None  # List of available client IDs
optimal_threshold = 0.475  # From LightGBM notebook results
model_type = None
model_loaded = False

# Paths configuration
MODEL_LOCAL_PATH = BASE_DIR / "src" / "models" / "lgbm_best_model.pkl"
FEATURES_PATH = BASE_DIR / "src" / "dataset" / "features_train_sample_1000.pkl"  # Use sample for Azure deployment
FEATURES_PATH_FALLBACK = BASE_DIR / "src" / "dataset" / "features_train.pkl"  # Fallback to full dataset
FEATURE_NAMES_PATH = BASE_DIR / "src" / "dataset" / "feature_names.json"
CLIENT_IDS_PATH = BASE_DIR / "src" / "dataset" / "client_ids_sample_1000.json"


def load_model_from_local() -> tuple:
    """Load model from local pickle file
    
    Returns:
        tuple: (model, model_type) or (None, None) if loading fails
    """
    try:
        if MODEL_LOCAL_PATH.exists():
            logger.info(f"Loading model from local file: {MODEL_LOCAL_PATH}")
            with open(MODEL_LOCAL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully from local file")
            return model, "local"
        else:
            logger.warning(f"Model file not found: {MODEL_LOCAL_PATH}")
            return None, None
    except Exception as e:
        logger.error(f"Error loading model from local file: {e}")
        return None, None


def load_feature_names_from_file() -> Optional[List[str]]:
    """Load feature names from JSON file"""
    try:
        if FEATURE_NAMES_PATH.exists():
            with open(FEATURE_NAMES_PATH, 'r') as f:
                data = json.load(f)
            feature_names = data.get("feature_names", [])
            logger.info(f"Loaded {len(feature_names)} feature names from file")
            return feature_names
        else:
            logger.warning(f"Feature names file not found: {FEATURE_NAMES_PATH}")
            # Try to load from feature_io utility
            try:
                feature_names = load_feature_names(input_dir=str(BASE_DIR / "data" / "processed"))
                logger.info(f"Loaded {len(feature_names)} feature names from utility")
                return feature_names
            except Exception as e:
                logger.error(f"Error loading feature names: {e}")
                return None
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")
        return None


def load_full_dataset() -> Optional[pd.DataFrame]:
    """Load sample dataset (1000 rows) for client ID lookup"""
    try:
        # Try sample dataset first (for Azure deployment)
        if FEATURES_PATH.exists():
            logger.info(f"Loading sample dataset from: {FEATURES_PATH}")
            with open(FEATURES_PATH, 'rb') as f:
                df = pickle.load(f)
            logger.info(f"Sample dataset loaded: {df.shape}")
        elif FEATURES_PATH_FALLBACK.exists():
            logger.warning(f"Sample dataset not found, using full dataset: {FEATURES_PATH_FALLBACK}")
            with open(FEATURES_PATH_FALLBACK, 'rb') as f:
                df = pickle.load(f)
            logger.info(f"Full dataset loaded: {df.shape}")
        else:
            logger.error(f"Neither sample nor full dataset found")
            return None
        
        # Set SK_ID_CURR as index for faster lookup
        if 'SK_ID_CURR' in df.columns:
            df = df.set_index('SK_ID_CURR')
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None


def load_client_ids() -> Optional[List[int]]:
    """Load list of available client IDs from JSON file"""
    try:
        if CLIENT_IDS_PATH.exists():
            with open(CLIENT_IDS_PATH, 'r') as f:
                data = json.load(f)
            client_ids = data.get("client_ids", [])
            logger.info(f"Loaded {len(client_ids)} client IDs from file")
            return client_ids
        else:
            logger.warning(f"Client IDs file not found: {CLIENT_IDS_PATH}")
            # Fallback: extract from dataset if available
            if full_dataset is not None:
                client_ids = sorted(full_dataset.index.unique().tolist())
                logger.info(f"Extracted {len(client_ids)} client IDs from dataset")
                return client_ids
            return None
    except Exception as e:
        logger.error(f"Error loading client IDs: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    global model, feature_names, full_dataset, client_ids, model_type, model_loaded
    
    logger.info("Starting up API...")
    
    # Load model from local file
    result = load_model_from_local()
    if result[0] is not None:
        model, model_type = result
        
        # Load feature names
        feature_names = load_feature_names_from_file()
        if feature_names is None:
            logger.warning("Feature names not loaded, will try to infer from model")
        
        # Load sample dataset for client ID lookup
        full_dataset = load_full_dataset()
        if full_dataset is None:
            logger.warning("Dataset not loaded, client ID lookup will not work")
        else:
            # Load client IDs list
            client_ids = load_client_ids()
            if client_ids is None:
                logger.warning("Client IDs not loaded, will extract from dataset if needed")
        
        model_loaded = True
        logger.info("API startup complete")
    else:
        logger.error("Failed to load model from local file")
        model_loaded = False
    
    yield
    
    # Shutdown (if needed)
    logger.info("Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring API",
    description="API for credit scoring predictions using LightGBM model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class ClientIdRequest(BaseModel):
    """Request model for client ID prediction"""
    client_id: int


class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    client_id: int
    probability: float
    prediction: int
    threshold: float
    recommendation: str


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str
    n_features: int
    optimal_threshold: float
    feature_names: List[str]
    model_loaded: bool


class ClientIdsResponse(BaseModel):
    """Response model for client IDs list"""
    client_ids: List[int]
    total: int


class ShapValueItem(BaseModel):
    """Model for a single SHAP value"""
    feature: str
    value: float
    importance: float  # Absolute value for sorting


class ShapResponse(BaseModel):
    """Response model for SHAP values"""
    client_id: int
    shap_values: List[ShapValueItem]
    top_features: int


def get_client_features(client_id: int) -> Optional[pd.DataFrame]:
    """Get features for a specific client ID"""
    if full_dataset is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset not loaded. Cannot retrieve client features."
        )
    
    if client_id not in full_dataset.index:
        raise HTTPException(
            status_code=404,
            detail=f"Client ID {client_id} not found in dataset"
        )
    
    # Get client features
    client_features = full_dataset.loc[[client_id]].copy()
    
    # Remove non-feature columns
    exclude_cols = ['TARGET', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
    for col in exclude_cols:
        if col in client_features.columns:
            client_features = client_features.drop(columns=[col])
    
    return client_features


def validate_and_prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare features for prediction"""
    if feature_names is None:
        # If feature names not available, use all columns except known exclusions
        exclude_cols = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
        available_features = [col for col in df.columns if col not in exclude_cols]
        logger.warning(f"Using {len(available_features)} features (feature names not loaded)")
        return df[available_features]
    
    # Check for missing features
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {list(missing_features)[:10]}..." if len(missing_features) > 10 else f"Missing features: {list(missing_features)}"
        )
    
    # Select and order features according to model
    features_df = df[feature_names].copy()
    
    # Handle missing values (fill with 0 or median)
    if features_df.isnull().any().any():
        logger.warning("Found missing values in features, filling with 0")
        features_df = features_df.fillna(0)
    
    return features_df


def make_prediction(features: pd.DataFrame) -> np.ndarray:
    """Make prediction using the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Get probabilities
        probabilities = model.predict_proba(features)[:, 1]
        return probabilities
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


def calculate_shap_values(features: pd.DataFrame, top_n: int = 20) -> List[Dict]:
    """Calculate SHAP values for a prediction
    
    Args:
        features: DataFrame with features for prediction
        top_n: Number of top features to return
        
    Returns:
        List of dictionaries with feature, value, and importance
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Create SHAP explainer (TreeExplainer works with tree-based models)
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features)
        
        # For binary classification, shap_values is a list [values_class_0, values_class_1]
        # We want the values for class 1 (default)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get values for positive class
        
        # Get feature names
        feature_list = features.columns.tolist()
        
        # Create list of (feature, shap_value, abs_value) tuples
        shap_list = [
            {
                "feature": feature_list[i],
                "value": float(shap_values[0, i]),
                "importance": float(abs(shap_values[0, i]))
            }
            for i in range(len(feature_list))
        ]
        
        # Sort by absolute value (importance) and get top N
        shap_list.sort(key=lambda x: x["importance"], reverse=True)
        top_shap = shap_list[:top_n]
        
        return top_shap
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"SHAP calculation error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_type": model_type if model_loaded else None
    }


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_type": model_type or "unknown",
        "n_features": len(feature_names) if feature_names else 0,
        "optimal_threshold": optimal_threshold,
        "feature_names": feature_names[:10] if feature_names else [],  # Return first 10 for brevity
        "model_loaded": model_loaded
    }


@app.get("/clients/ids", response_model=ClientIdsResponse)
async def get_client_ids():
    """Get list of available client IDs"""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    if client_ids is None:
        # Fallback: extract from dataset if available
        if full_dataset is not None:
            ids = sorted(full_dataset.index.unique().tolist())
            return {
                "client_ids": ids,
                "total": len(ids)
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="Client IDs not available"
            )
    
    return {
        "client_ids": client_ids,
        "total": len(client_ids)
    }


@app.post("/predict/client_id", response_model=PredictionResponse)
async def predict_by_client_id(request: ClientIdRequest):
    """Make prediction for a specific client ID"""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Get client features
        client_features = get_client_features(request.client_id)
        
        # Validate and prepare features
        features = validate_and_prepare_features(client_features)
        
        # Make prediction
        probability = make_prediction(features)[0]
        
        # Apply threshold
        prediction = 1 if probability >= optimal_threshold else 0
        
        # Generate recommendation
        if prediction == 1:
            recommendation = "⚠️ Risque de défaut élevé - Prêt non recommandé"
        else:
            recommendation = "✅ Risque de défaut faible - Prêt recommandé"
        
        return {
            "client_id": request.client_id,
            "probability": float(probability),
            "prediction": int(prediction),
            "threshold": optimal_threshold,
            "recommendation": recommendation
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_by_client_id: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/predict/client_id/shap", response_model=ShapResponse)
async def predict_by_client_id_shap(
    request: ClientIdRequest,
    top_n: int = Query(20, ge=1, le=50, description="Number of top features to return")
):
    """Get SHAP values for a specific client ID prediction"""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Get client features
        client_features = get_client_features(request.client_id)
        
        # Validate and prepare features
        features = validate_and_prepare_features(client_features)
        
        # Calculate SHAP values
        shap_list = calculate_shap_values(features, top_n=top_n)
        
        # Convert to ShapValueItem models
        shap_items = [
            ShapValueItem(
                feature=item["feature"],
                value=item["value"],
                importance=item["importance"]
            )
            for item in shap_list
        ]
        
        return {
            "client_id": request.client_id,
            "shap_values": shap_items,
            "top_features": len(shap_items)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_by_client_id_shap: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
