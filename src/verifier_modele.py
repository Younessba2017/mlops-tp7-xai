import mlflow
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xai_commun import CAS, VERSION, Xte, m, yte

run = mlflow.get_run(VERSION.run_id)
pred = m.predict(Xte)
proba = m.predict_proba(Xte)[:, 1]

print("Modele     : fraude, version", VERSION.version, "| run", VERSION.run_id[:8])
print("Classe     :", type(m).__name__)
print("Parametres :", {k: run.data.params[k] for k in ["n_estimators", "max_depth"]})
print(f"F1         : {f1_score(yte, pred):.4f} | F1 enregistre : {run.data.metrics['f1']:.4f}")
print(f"Precision  : {precision_score(yte, pred):.4f}")
print(f"Rappel     : {recall_score(yte, pred):.4f}")
print(f"AUC        : {roc_auc_score(yte, proba):.4f}")
print("Matrice    :", confusion_matrix(yte, pred).tolist())
print(f"Mme Benali : P(fraude) = {m.predict_proba(CAS)[0, 1]:.4f}")