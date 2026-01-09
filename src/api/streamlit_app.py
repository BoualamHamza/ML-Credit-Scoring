"""
Streamlit interface for Credit Scoring API

This application provides a user-friendly interface to interact with the
Credit Scoring FastAPI backend for making predictions.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from typing import Optional, Dict

# Configuration
API_URL = "http://localhost:8000"
API_TIMEOUT = 30


def check_api_health() -> bool:
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        return False


def get_model_info() -> Optional[Dict]:
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


def predict_by_client_id(client_id: int) -> Optional[Dict]:
    """Make prediction for a client ID"""
    try:
        response = requests.post(
            f"{API_URL}/predict/client_id",
            json={"client_id": client_id},
            timeout=API_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.error(f"Client ID {client_id} not found in dataset")
            return None
        else:
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"API Error: {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please make sure the API is running on http://localhost:8000")
        return None
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None


def display_prediction_result(result: Dict):
    """Display prediction result with visualizations"""
    client_id = result.get("client_id", "N/A")
    probability = result.get("probability", 0.0)
    prediction = result.get("prediction", 0)
    threshold = result.get("threshold", 0.475)
    recommendation = result.get("recommendation", "")
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Client ID", client_id)
    
    with col2:
        st.metric("Probabilité de défaut", f"{probability:.2%}")
    
    with col3:
        prediction_label = "⚠️ Risque élevé" if prediction == 1 else "✅ Risque faible"
        st.metric("Prédiction", prediction_label)
    
    # Recommendation
    st.info(recommendation)
    
    # Visualization
    fig = go.Figure()
    
    # Add probability bar
    color = "red" if prediction == 1 else "green"
    fig.add_trace(go.Bar(
        x=["Probabilité de défaut"],
        y=[probability],
        marker_color=color,
        text=[f"{probability:.2%}"],
        textposition="auto",
        name="Probabilité"
    ))
    
    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Seuil optimal ({threshold:.3f})",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Probabilité de défaut",
        yaxis_title="Probabilité",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed information
    with st.expander("📊 Détails de la prédiction"):
        st.json(result)


def predict_by_client_id_ui():
    """UI for prediction by client ID"""
    st.header("🔍 Prédiction par ID Client")
    
    st.markdown("""
    Entrez l'ID du client (SK_ID_CURR) pour obtenir une prédiction de risque de défaut.
    """)
    
    # Input for client ID
    client_id = st.number_input(
        "ID Client (SK_ID_CURR)",
        min_value=100000,
        max_value=999999999,
        value=100001,
        step=1,
        help="Entrez l'identifiant unique du client"
    )
    
    if st.button("🔮 Prédire", type="primary"):
        with st.spinner("Calcul de la prédiction en cours..."):
            result = predict_by_client_id(client_id)
            
            if result:
                st.success("✅ Prédiction effectuée avec succès")
                display_prediction_result(result)


def main():
    """Main application"""
    st.set_page_config(
        page_title="Credit Scoring - Prédiction de Défaut",
        page_icon="🏦",
        layout="wide"
    )
    
    # Title
    st.title("🏦 Système de Scoring de Crédit")
    st.markdown("---")
    
    # Check API health
    if not check_api_health():
        st.error("""
        ⚠️ **L'API n'est pas accessible**
        
        Veuillez démarrer l'API FastAPI en exécutant:
        ```bash
        uvicorn src.api.api:app --reload
        ```
        ou
        ```bash
        python -m src.api.api
        ```
        """)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model info
        model_info = get_model_info()
        if model_info:
            st.success("✅ API connectée")
            st.markdown(f"**Type de modèle:** {model_info.get('model_type', 'N/A')}")
            st.markdown(f"**Nombre de features:** {model_info.get('n_features', 'N/A')}")
            st.markdown(f"**Seuil optimal:** {model_info.get('optimal_threshold', 'N/A')}")
        else:
            st.warning("⚠️ Impossible de récupérer les informations du modèle")
        
        st.markdown("---")
    
    # Main content - only client ID prediction
    predict_by_client_id_ui()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>Credit Scoring API v1.0.0 | Powered by LightGBM</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
