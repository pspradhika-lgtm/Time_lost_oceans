import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tensorflow.keras import layers, models

st.title("🌊 Time-Lost Oceans: Predict Sunken Civilizations")

# -----------------------------
# Load datasets from repo
# -----------------------------
X_images = np.load('ocean_images.npy')
X_tabular = pd.read_csv('historical_data.csv')

st.write("Dataset preview:")
st.dataframe(X_tabular.head())

# Features and labels
y = X_tabular['label'].values
X_tabular = X_tabular.drop(columns=['label'])

# Scale tabular data
scaler = StandardScaler()
X_tab_scaled = scaler.fit_transform(X_tabular)

# -----------------------------
# CNN for ocean images
# -----------------------------
cnn_model = models.Sequential([
    layers.Input(shape=(64,64,1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu')
])

st.write("Training CNN embeddings... (few seconds)")
X_img_emb = cnn_model(X_images).numpy()

# -----------------------------
# Combine CNN + Tabular
# -----------------------------
X_combined = np.hstack([X_img_emb, X_tab_scaled])

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_combined, y)
st.success("✅ Model trained successfully!")

# -----------------------------
# Sidebar inputs for prediction
# -----------------------------
st.sidebar.header("Predict a new site")
moon_phase = st.sidebar.slider("Moon Phase (0-1)", 0.0, 1.0, 0.5)
solstice_distance = st.sidebar.slider("Solstice Distance (deg)", 0, 180, 90)
known_site_distance = st.sidebar.slider("Distance to known site", 0, 50, 25)

# Random ocean map for demo
user_image = np.random.rand(1,64,64,1)

# Scale tabular input
user_tab = np.array([[moon_phase, solstice_distance, known_site_distance]])
user_tab_scaled = scaler.transform(user_tab)

# CNN embedding
user_emb = cnn_model(user_image).numpy()

# Combine and predict
user_combined = np.hstack([user_emb, user_tab_scaled])
pred_prob = xgb_model.predict_proba(user_combined)[0][1]
pred_label = xgb_model.predict(user_combined)[0]

st.write(f"Predicted Outcome: {'✅ Sunken Site Likely' if pred_label==1 else '❌ No Site'}")
st.write(f"Probability: {pred_prob*100:.2f}%")
