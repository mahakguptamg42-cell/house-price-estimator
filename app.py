import os
import xgboost as xgb
import streamlit as st

@st.cache_resource
def load_trained_model():
    model_path = 'best_indian_house_model.json'
    if os.path.exists(model_path):
        try:
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"❌ Error loading model JSON: {e}")
            return None
    else:
        st.error("⚠️ Model file `best_indian_house_model.json` not found in repository!")
        return None

model = load_trained_model()
