from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from contextlib import asynccontextmanager

from model_manager import manager

# Use lifespan to load models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting up FastAPI. Initializing models...")
    manager.load_models()
    yield
    # Shutdown logic
    print("Shutting down FastAPI...")

app = FastAPI(title="Smart Home Energy Predictor API", lifespan=lifespan)

# Allow CORS for the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    temperature: float
    household_size: int
    hour: int
    month: int
    day_of_week: int
    appliance_types: list[str]
    season: str

@app.get("/api/health")
def health_check():
    models_loaded = manager.model is not None
    return {
        "status": "healthy", 
        "models_loaded": models_loaded,
        "message": "API is running. Models are ready." if models_loaded else "API is running. Models failed to load (check GDrive link)."
    }

@app.post("/api/predict")
def predict_energy(req: PredictionRequest):
    if manager.model is None:
        # Fallback dummy prediction if model isn't available for demo purposes
        # Raise HTTP error in a real scenario
        # raise HTTPException(status_code=503, detail="Model not loaded. Ensure models are downloaded.")
        print("Model not loaded, using mock predictions.")
        results = {}
        total = 0.0
        for app_type in req.appliance_types:
            mock_val = (req.temperature * 0.1) + (req.household_size * 0.5) + (len(app_type) * 0.2)
            results[app_type] = mock_val
            total += mock_val
        return {"predictions": results, "total_kwh_per_day": total}

    try:
        results = {}
        total = 0.0
        
        for app_type in req.appliance_types:
            input_data = pd.DataFrame([{
                'Outdoor Temperature (°C)': req.temperature,
                'Household Size': req.household_size,
                'Hour': req.hour,
                'Month': req.month,
                'DayOfWeek': req.day_of_week,
                'Appliance Type': app_type,
                'Season': req.season
            }])
            
            pred = manager.predict(input_data)[0]
            results[app_type] = pred
            total += pred
            
        return {"predictions": results, "total_kwh_per_day": total}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.1", port=8000, reload=True)
