import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def initialize_mlflow(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def start_run(run_name):
    return mlflow.start_run(run_name=run_name)

def log_parameters(params):
    mlflow.log_params(params)

def log_metrics(metrics):
    mlflow.log_metrics(metrics)

def log_artifacts(local_dir, artifact_path=None):
    if artifact_path:
        mlflow.log_artifacts(local_dir, artifact_path)
    else:
        mlflow.log_artifacts(local_dir)

def log_artifact_file(local_path, artifact_path=None):
    if artifact_path:
        mlflow.log_artifact(local_path, artifact_path)
    else:
        mlflow.log_artifact(local_path)

def log_model(model, artifact_path, X_example, registered_model_name=None):
    # Only use predictive columns for signature, omit target or identifiers
    signature = infer_signature(X_example, model.predict(X_example))
    if registered_model_name:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            registered_model_name=registered_model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
    else:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

def add_tags(tags):
    mlflow.set_tags(tags)
