from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from train import load_model, load_data
from catboost import Pool
import os

app = FastAPI()

class Diamond(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

# This would be set after training the model for the first time or loading a pre-trained one.
MODEL_PATH = 'test_model'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None

@app.post("/predict/")
async def predict(diamond: Diamond):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not trained yet. Train the model first.")
    
    data = pd.DataFrame([diamond.model_dump()])
    pool = Pool(data, cat_features=['cut', 'color', 'clarity'])
    prediction = model.predict(pool)
    return {"prediction": prediction[0]}

@app.post("/predict_csv/", response_model=str, responses={200: {"content": {"text/csv": {}}}})
async def predict_csv(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not trained yet. Train the model first.")
    
    # Read the uploaded file into a DataFrame
    df = pd.read_csv(file.file)

    # Ensure columns are present
    expected_cols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    for col in expected_cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing column: {col}")

    # Create a pool and make predictions
    pool = Pool(df, cat_features=['cut', 'color', 'clarity'])
    predictions = model.predict(pool)

    # Add predictions to the dataframe
    df['prediction'] = predictions

    # Convert the DataFrame to a CSV string and return it
    csv_output = df.to_csv(index=False)
    return csv_output

@app.post("/train/")
async def train(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df.to_csv("tmp_data.csv", index=False)

    # Mimicking argparse for calling the main function
    class Args:
        data_path = "tmp_data.csv"
        depth = 8
        learning_rate = 0.03
        loss_function = 'RMSE'
        iterations = 1000
        verbose = 200
        task_type = 'CPU'
        train_save_path = None
        test_save_path = None
        output_path = None
        format = 'csv'
        save_model_path = MODEL_PATH
        load_model_path = None

    from train import main
    main(Args())
    return {"detail": "Training completed and model updated."}

