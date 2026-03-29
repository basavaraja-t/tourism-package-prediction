
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.title("Tourism Package Prediction")

repo_id = "basavarajat/tourism-package-model"

# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id=repo_id,
    filename="tourism_model.pkl"
)

model = joblib.load(model_path)

st.write("Enter customer details:")

age = st.number_input("Age", 18, 100)
city_tier = st.selectbox("City Tier", [1, 2, 3])
monthly_income = st.number_input("Monthly Income", 1000, 1000000)

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Age": age,
        "CityTier": city_tier,
        "MonthlyIncome": monthly_income
    }])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Customer is likely to BUY the package")
    else:
        st.error("Customer is NOT likely to buy")
