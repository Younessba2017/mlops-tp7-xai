import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from xai_commun import CAS, Xte, m

explicateur = shap.TreeExplainer(m)
probabilites = m.predict_proba(Xte)[:, 1]
indice = Xte.index[probabilites.argmax()]

cas_a_expliquer = [
    ("benali", CAS),
    (f"test_{indice}", Xte.loc[[indice]]),
]

for nom, transaction in cas_a_expliquer:
    explication = explicateur(transaction)[0, :, 1]

    print(f"\n--- {nom} :", transaction.iloc[0].to_dict())
    print(f"Valeur de base : {float(explication.base_values):.4f}")

    contributions = sorted(
        zip(transaction.columns, explication.values),
        key=lambda element: -abs(element[1]),
    )
    for variable, contribution in contributions:
        print(f"  {variable:11s} {contribution:+.4f}")

    total = float(explication.base_values + explication.values.sum())
    prediction = m.predict_proba(transaction)[0, 1]
    print(f"Base + somme : {total:.4f} | Prediction : {prediction:.4f}")

    shap.plots.waterfall(explication, show=False)
    plt.tight_layout()
    plt.savefig(f"reports/shap_{nom}.png", dpi=120)
    plt.close()