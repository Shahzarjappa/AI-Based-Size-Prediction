import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Size Recommendation Engine", page_icon="👚", layout="centered")

# --- CUSTOM CSS FOR BRANDING (Sapphire/Khaadi Vibe) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #1a4731; color: white; border-radius: 8px;
        width: 100%; font-size: 18px; font-weight: bold; height: 50px;
    }
    .stButton>button:hover { background-color: #113021; color: white; }
    .result-box {
        background-color: #e2f0d9; border-left: 6px solid #385723;
        padding: 20px; border-radius: 8px; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_assets():
    model = joblib.load('size_prediction_model.pkl')
    encoder = joblib.load('size_encoder.pkl')
    features = joblib.load('model_features.pkl')
    return model, encoder, features

try:
    model, le, model_features = load_assets()
except:
    st.error("Error: Please make sure all 3 model files (.pkl) are in this exact folder!")
    st.stop()

# Title
st.markdown("<h2 style='text-align: center; color: #1a4731;'>Smart Size Recommendation Widget</h2>", unsafe_allow_html=True)
st.write("Enter customer body metrics and product specifications to predict the perfect fitting size.")
st.markdown("---")

# --- UI LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 Customer Body Profile")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 15, 80, 28)
    height = st.number_input("Height (cm)", min_value=120.0, max_value=210.0, value=165.0, step=0.5)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=65.0, step=0.5)
    
    # Auto Calculated BMI
    bmi = round(weight / ((height / 100) ** 2), 1)
    st.info(f"Calculated BMI: **{bmi}**")
    
    body_type = st.selectbox("Body Type", ["slim", "regular", "heavy"])

with col2:
    st.markdown("#### 🛍️ Product Catalog Spec")
    brand = st.selectbox("Brand / Retailer", ["Sapphire", "Khaadi", "Limelight", "Ideas", "Nishat Linen", "Ethnic"])
    category = st.selectbox("Product Category", ["Kurta", "Shirt", "Pants", "Trouser", "Eastern Suit"])
    fit_type = st.selectbox("Fit Silhouette", ["slim", "regular", "loose"])
    fabric_type = st.selectbox("Fabric Type", ["Lawn", "Cotton", "Silk", "Linen", "Cambric", "Khaddar"])

st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("Find My Perfect Fit ✨"):
    
    # 1. Recreating Gender-Specific Anthropometrics Formulas (Live)
    if gender == "Male":
        est_user_chest = (weight * 0.42) + (height * 0.11)
        est_user_waist = (weight * 0.45) + (height * 0.07)
    else:
        est_user_chest = (weight * 0.38) + (height * 0.09)
        est_user_waist = (weight * 0.40) + (height * 0.05)
        
    # 2. Recreating Cross-Interaction Features
    brand_fit = f"{brand}_{fit_type}"
    category_fit = f"{category}_{fit_type}"
    
    # 3. Create Input Dictionary
    input_data = {
        'height_cm': height, 'weight_kg': weight, 'bmi': bmi, 'age': age,
        'est_user_chest': est_user_chest, 'est_user_waist': est_user_waist,
        f'gender_{gender}': 1,
        f'body_type_{body_type}': 1,
        f'brand_{brand}': 1,
        f'category_{category}': 1,
        f'fit_type_{fit_type}': 1,
        f'fabric_type_{fabric_type}': 1,
        f'brand_fit_{brand_fit}': 1,
        f'category_fit_{category_fit}': 1
    }
    
    # 4. Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Align with training features matrix
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[model_features] # Maintain column order
    
    # 5. Run Model Prediction
    prediction_numeric = model.predict(input_df)[0]
    recommended_size = le.inverse_transform([prediction_numeric])[0]
    
    # 6. Display Result Box
    st.markdown(f"""
        <div class="result-box">
            <h3 style="color: #385723; margin: 0;">RECOMMENDED SIZE</h3>
            <h1 style="color: #1a4731; font-size: 50px; margin: 10px 0;">{recommended_size}</h1>
            <p style="color: #555; margin: 0;">This size matches the best geometric proportions for your body type with <b>{brand}</b> specifications.</p>
        </div>
    """, unsafe_allow_html=True)