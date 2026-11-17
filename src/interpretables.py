import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from xai_commun import CAS, X, Xtr, Xte, m, ytr, yte

resultats = {}


def evaluer(nom, modele, donnees):
    predictions = modele.predict(donnees)
    mesures = {
        "f1": round(f1_score(yte, predictions), 4),
        "fraudes_detectees": int(
            ((predictions == 1) & (yte == 1)).sum()
        ),
    }
    resultats[nom] = mesures
    print(
        f"{nom:42s} F1 = {mesures['f1']:.4f}"
        f" | fraudes detectees : {mesures['fraudes_detectees']}"
        f" / {int(yte.sum())}"
    )


def transformer_montant(donnees):
    donnees = donnees.copy()
    donnees["log_montant"] = np.log10(donnees.pop("montant"))
    return donnees


evaluer("foret fraude@production", m, Xte)

logistique_brut = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
).fit(Xtr, ytr)
evaluer("logistique, montant brut", logistique_brut, Xte)

Xtr_log = transformer_montant(Xtr)
Xte_log = transformer_montant(Xte)

logistique_log = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
).fit(Xtr_log, ytr)
evaluer("logistique, log10(montant)", logistique_log, Xte_log)

coefficients = dict(zip(
    Xtr_log.columns,
    logistique_log[-1].coef_[0].round(2).tolist(),
))
print("\nCoefficients standardises :", coefficients)

multiplicateurs = dict(zip(
    Xtr_log.columns,
    np.exp(logistique_log[-1].coef_[0] / logistique_log[0].scale_),
))
print(
    f"Cote x {multiplicateurs['anciennete']:.3f} par mois d'anciennete"
    f" ; x {multiplicateurs['freq_24h']:.2f} par transaction en 24 h"
)

for profondeur in [2, 3, 6, None]:
    arbre = DecisionTreeClassifier(
        max_depth=profondeur, random_state=0
    ).fit(Xtr, ytr)
    nom = (
        f"arbre profondeur {profondeur}"
        f" ({arbre.get_n_leaves()} feuilles)"
    )
    evaluer(nom, arbre, Xte)

arbre_court = DecisionTreeClassifier(
    max_depth=3, random_state=0
).fit(Xtr, ytr)

print("\nArbre de profondeur 3")
print(export_text(
    arbre_court, feature_names=list(X.columns), decimals=1
))
print(
    "Mme Benali : feuille/noeud",
    arbre_court.apply(CAS)[0],
    "| P(fraude) =",
    round(arbre_court.predict_proba(CAS)[0, 1], 3),
)

with open("reports/interpretables.json", "w", encoding="utf-8") as fichier:
    json.dump(resultats, fichier, indent=2)