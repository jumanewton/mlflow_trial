from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
from prometheus_fastapi_instrumentator import Instrumentator
from mlflow.tracking import MlflowClient

app = FastAPI()

# Instrument Prometheus
Instrumentator().instrument(app).expose(app)

# MLflow Tracking URI (use container network name, not localhost)
mlflow.set_tracking_uri("http://mlflow:5000")
# Set the experiment name
mlflow.set_experiment("Iris_Experiment")

# Dynamically fetch latest registered model version
model_name = "IrisModel"
client = MlflowClient()

# Get all registered versions
versions = [int(mv.version) for mv in client.search_model_versions(f"name='{model_name}'")]

if not versions:
    raise Exception(f"No registered model versions found for {model_name}")

# Select the latest version
latest_version = max(versions)

# Load the latest version
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")

# Request schema
class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: Features):
    data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}
