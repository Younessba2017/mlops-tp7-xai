import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from xai_commun import Xte, m

echantillon = Xte.sample(400, random_state=0)
explicateur = shap.TreeExplainer(m)
contributions = explicateur(echantillon)[:, :, 1]  # Classe 1 = fraude

importance = pd.Series(
    np.abs(contributions.values).mean(axis=0),
    index=echantillon.columns,
).sort_values(ascending=False)

print(f"Valeur de base : {contributions.base_values[0]:.4f}")
print(importance.round(4).to_string())
importance.round(4).to_json("reports/shap_importance.json")

shap.plots.beeswarm(contributions, show=False)
plt.tight_layout()
plt.savefig("reports/shap_global.png", dpi=120)
plt.close()

plt.scatter(
    echantillon["montant"],
    contributions[:, "montant"].values,
    c=echantillon["nuit"],
    cmap="coolwarm",
    s=12,
)
plt.xscale("log")
plt.xlabel("Montant (MAD)")
plt.ylabel("Contribution SHAP du montant")
plt.colorbar(label="nuit")
plt.tight_layout()
plt.savefig("reports/shap_montant.png", dpi=120)
plt.close()

print("Graphiques enregistres dans reports/")