import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load sample data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Correct MLflow tracking setup for Docker
mlflow.set_tracking_uri("http://mlflow:5000")  # Use service name, not localhost
mlflow.set_experiment("Iris_Experiment")

# Train model and track with MLflow
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, "model", registered_model_name="IrisModel")
    mlflow.log_metric("accuracy", model.score(X_test, y_test))

print("✅ Model logged and registered successfully.")
