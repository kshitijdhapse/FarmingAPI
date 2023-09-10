import uvicorn
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pickled model
with open("crop_predictor.pkl", "rb") as f:
    crop_predictor = pickle.load(f)

@app.get('/')
def root():
    return {'message': 'Welcome to the Crop Prediction API'}

@app.post('/predict')
def predict_crop(N: float, P: float, K: float, temp: float, hum: float, pH: float, rain: float):
    """Route to make predictions using the model."""
    try:
        # Access data attributes directly from the request
        prediction = crop_predictor.predict([[N, P, K, temp, hum, pH, rain]])
        return {'prediction': prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
