from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Load saved objects
model = joblib.load('linear_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
feature_names = joblib.load('feature_names.pkl')

app = FastAPI(title="Laptop Price Prediction API")

# Input schema - match your training columns
class LaptopInput(BaseModel):
    Brand: str
    Processor_Brand: str
    Processor_Series: str
    Processor_Generation: str
    Ram_GB: float
    Ram_Type: str
    Storage_Type: str
    Storage_GB_clean: float
    Stars: float
    Num_Ratings: int
    Num_Reviews: int
    discount: float

@app.get("/")
def home():
    return {"message": "Laptop Price Prediction API running!"}

@app.post("/predict")
def predict_price(data: LaptopInput):
    # Convert input JSON to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Ensure numerical columns are float
    num_cols = ['Ram_GB','Storage_GB_clean','Stars','Num_Ratings','Num_Reviews','discount']
    input_df[num_cols] = input_df[num_cols].astype(float)

    # Transform input using saved ColumnTransformer
    X_transformed = preprocessor.transform(input_df)

    # Convert transformed input to DataFrame with correct feature names
    X_df = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed,'toarray') else X_transformed,
                        columns=feature_names)

    # Ensure all columns exist (handle missing dummy columns)
    for col in feature_names:
        if col not in X_df.columns:
            X_df[col] = 0
    X_df = X_df[feature_names]

    # Predict price
    prediction = model.predict(X_df)

    return {"Predicted_Price": float(prediction[0])}
