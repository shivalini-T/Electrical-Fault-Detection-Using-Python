
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Electrical Fault Detection System",
    page_icon="⚡",
    layout="centered"
)

# Title
st.title("⚡ Electrical Fault Detection System")
st.write("Machine Learning Based Electrical Fault Detection")

st.divider()

# Load dataset
@st.cache_resource
def train_model():
    data = pd.read_csv("dataset/electrical_data.csv")

    X = data[["voltage", "current"]]
    y = data["fault_type"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)
    return model

model = train_model()

# User input
st.subheader("Enter Electrical Readings")

voltage = st.number_input(
    "Voltage (V)",
    min_value=0.0,
    value=230.0,
    step=1.0
)

current = st.number_input(
    "Current (A)",
    min_value=0.0,
    value=5.0,
    step=1.0
)

# Prediction
if st.button("Detect Fault", type="primary"):
    new_data = pd.DataFrame({
        "voltage": [voltage],
        "current": [current]
    })

    prediction = model.predict(new_data)[0]

    st.subheader("Fault Detection Result")

    if prediction == "Normal":
        st.success("⚡ NORMAL")
    else:
        st.error("⚠️ " + prediction.upper())

    st.write("Voltage:", voltage, "V")
    st.write("Current:", current, "A")

st.divider()

st.caption(
    "Educational prototype using a small synthetic dataset. "
    "Not intended for real-world electrical protection."
)
