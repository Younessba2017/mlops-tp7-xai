import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from xai_commun import CAS, X, Xtr, m


def predire(valeurs):
    # Conserver les noms des colonnes attendus par le modele.
    transactions = pd.DataFrame(valeurs, columns=X.columns)
    return m.predict_proba(transactions)


def expliquer(graine, nombre_voisins):
    explicateur = LimeTabularExplainer(
        Xtr.values,
        feature_names=list(X.columns),
        class_names=["normal", "fraude"],
        mode="classification",
        random_state=graine,
    )
    return explicateur.explain_instance(
        CAS.values[0],
        predire,
        num_features=4,
        num_samples=nombre_voisins,
    )


reference = expliquer(0, 2000)
print("Regles :", [
    (regle, round(poids, 3))
    for regle, poids in reference.as_list()
])

reference.save_to_file("reports/lime_benali.html")
figure = reference.as_pyplot_figure()
figure.tight_layout()
figure.savefig("reports/lime_benali.png", dpi=120)
plt.close(figure)

for nombre_voisins in [500, 2000]:
    print(f"\n--- {nombre_voisins} voisins")
    print("Graine  montant     nuit  anciennete  freq_24h")

    for graine in range(5):
        explication = expliquer(graine, nombre_voisins)
        poids = dict(explication.local_exp[1])

        print(
            f"{graine:6d}"
            f"  {poids[0]:+.3f}"
            f"  {poids[3]:+.3f}"
            f"      {poids[1]:+.3f}"
            f"     {poids[2]:+.3f}"
        )