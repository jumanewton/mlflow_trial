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

# Dynamically fetch latest Production model
model_name = "IrisModel"
client = MlflowClient()
latest_production_version = None

# Look for latest production model version
for mv in client.search_model_versions(f"name='{model_name}'"):
    if mv.current_stage == "Production":
        latest_production_version = mv.version
        break

if latest_production_version is None:
    raise Exception(f"No production model version found for {model_name}")

model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_production_version}")

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
