from pathlib import Path

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

run_id = Path("reports/run_xai.txt").read_text(
    encoding="utf-8"
).strip()

run = client.get_run(run_id)
nom = run.data.tags["modele"]
version = run.data.tags["modele_version"]

card = Path("MODEL_CARD.md").read_text(encoding="utf-8")

with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact("MODEL_CARD.md")

client.update_model_version(
    nom, version, description=card
)
client.set_model_version_tag(
    nom,
    version,
    "model_card",
    f"MODEL_CARD.md, run {run_id}",
)

print(f"Model Card attachee a {nom} v{version}")
print("Run des explications :", run_id)