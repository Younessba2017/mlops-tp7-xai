import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from xai_commun import X, Xte, m, yte

resultat = permutation_importance(
    m, Xte, yte, scoring="f1", n_repeats=10, random_state=0
)

tableau = pd.DataFrame(
    {
        "MDI (foret)": m.feature_importances_,
        "permutation": resultat.importances_mean,
        "ecart-type": resultat.importances_std,
    },
    index=X.columns,
).sort_values("permutation", ascending=False).round(3)

print(tableau.to_string())
tableau.to_json("reports/importance.json")

ax = tableau["permutation"].plot.barh(xerr=tableau["ecart-type"], color="#4592EF")
ax.set_xlabel("Baisse du F1 quand la colonne est melangee")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("reports/permutation.png", dpi=120)
plt.close()