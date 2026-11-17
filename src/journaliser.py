import hashlib
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score

from xai_commun import ALIAS, NOM, VERSION, Xte, m, yte

empreinte = hashlib.md5(
    Path("data/transactions.csv").read_bytes()
).hexdigest()

predictions = m.predict(Xte)
mlflow.set_experiment("tp7-explicabilite")

with mlflow.start_run(
    run_name=f"xai-{NOM}-v{VERSION.version}"
) as run:
    mlflow.set_tags({
        "modele": NOM,
        "alias": ALIAS,
        "modele_version": VERSION.version,
        "modele_run_id": VERSION.run_id,
        "donnees": "data/transactions.csv",
        "donnees_md5": empreinte,
        "methodes": "permutation ; SHAP TreeExplainer ; LIME",
    })

    mlflow.log_params({
        "shap_echantillon": 400,
        "lime_num_samples": 2000,
        "lime_random_state": 0,
        "permutation_n_repeats": 10,
    })

    mlflow.log_metrics({
        "f1": f1_score(yte, predictions),
        "precision": precision_score(yte, predictions),
        "recall": recall_score(yte, predictions),
    })

    for fichier in sorted(Path("reports").glob("*.*")):
        if fichier.suffix in (".png", ".html", ".json"):
            mlflow.log_artifact(
                str(fichier), artifact_path="explications"
            )

    identifiant_run = run.info.run_id
    Path("reports/run_xai.txt").write_text(
        identifiant_run, encoding="utf-8"
    )

MlflowClient().set_model_version_tag(
    NOM,
    VERSION.version,
    "explications_run",
    identifiant_run,
)

print("Run des explications :", identifiant_run)
print(f"Explications du modele {NOM} v{VERSION.version} archivees")
print("Empreinte des donnees :", empreinte)