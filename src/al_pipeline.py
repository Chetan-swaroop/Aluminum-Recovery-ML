"""
ML-Optimized Aluminium Recovery for Al-Air Batteries
Stage 3 (purification route + cost) and Stage 4 (battery performance)

NOTE: The data in this script is SIMULATED from trends reported in the
literature (see the 'ASSUMPTIONS' comments). It is not measured data.
Run:  pip install numpy pandas scikit-learn matplotlib
      python al_pipeline.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

SEED = 42
rng = np.random.default_rng(SEED)
ROUTES = ["Remelting", "Fluxing", "Electrorefining"]
REQUIRED_PURITY = 99.9          # % Al needed for the battery anode

# ------------------------------------------------------------------
# STAGE 3 : simulated dataset  (500 records)
# ------------------------------------------------------------------
# ASSUMPTIONS (fraction of each impurity removed by each route)
#   Remelting + dross removal : removes oxides, almost no dissolved Fe/Si/Cu
#   Fluxing                   : removes part of Si, little Fe and Cu
#   Electrorefining           : removes almost everything (3-layer cell)
REMOVAL = {                      #    Fe          Si          Cu
    "Remelting":       [(0.00, 0.05), (0.00, 0.08), (0.00, 0.05)],
    "Fluxing":         [(0.10, 0.30), (0.40, 0.70), (0.05, 0.20)],
    "Electrorefining": [(0.995, 0.9995), (0.995, 0.9995), (0.995, 0.9995)],
}
# ASSUMPTIONS (USD per tonne, kWh per tonne)
COST   = {"Remelting": (180, 40), "Fluxing": (380, 60), "Electrorefining": (1500, 150)}
ENERGY = {"Remelting": (750, 80), "Fluxing": (1100, 100), "Electrorefining": (14000, 800)}

def simulate_stage3(fe, si, cu, route, noise=True):
    """returns purity (%), cost (USD/t), energy (kWh/t) for one item and route"""
    imp = np.array([fe, si, cu])
    if noise:
        frac = np.array([rng.uniform(*r) for r in REMOVAL[route]])
    else:
        frac = np.array([np.mean(r) for r in REMOVAL[route]])
    left = imp * (1 - frac)
    other = 0.05 if noise is False else rng.uniform(0.02, 0.10)   # Mg, Mn, Zn etc.
    if route == "Electrorefining":
        other *= 0.02
    purity = 100 - left.sum() - other
    total = imp.sum()
    c_mu, c_sd = COST[route]
    e_mu, e_sd = ENERGY[route]
    cost   = c_mu * (1 + 0.15 * (total - 0.8)) + (rng.normal(0, c_sd) if noise else 0)
    energy = e_mu * (1 + 0.10 * (total - 0.8)) + (rng.normal(0, e_sd) if noise else 0)
    return purity, cost, energy

def make_stage3(n=500):
    rows = []
    for _ in range(n):
        # impurity levels in wt% (skewed so some items are very clean, like foil)
        fe = np.clip(rng.lognormal(np.log(0.30), 0.55), 0.05, 1.2)
        si = np.clip(rng.lognormal(np.log(0.20), 0.55), 0.05, 0.9)
        cu = np.clip(rng.lognormal(np.log(0.12), 0.60), 0.02, 0.6)
        route = ROUTES[rng.integers(0, 3)]
        p, c, e = simulate_stage3(fe, si, cu, route)
        rows.append([fe, si, cu, route, p, c, e])
    return pd.DataFrame(rows, columns=["Fe", "Si", "Cu", "Route", "Purity", "Cost", "Energy"])

def route_onehot(df):
    X = df[["Fe", "Si", "Cu"]].copy()
    for r in ROUTES:
        X["Route_" + r] = (df["Route"] == r).astype(int)
    return X

df3 = make_stage3(500)
X3 = route_onehot(df3)
Xtr, Xte, idx_tr, idx_te = train_test_split(X3, df3.index, test_size=0.2, random_state=SEED)

print("=" * 60)
print("STAGE 3 : Purification route and cost prediction")
print("=" * 60)
print(f"Records: {len(df3)}   Train: {len(Xtr)}   Test: {len(Xte)}")
models3, res3 = {}, {}
for target, unit in [("Purity", "%"), ("Cost", "USD/t"), ("Energy", "kWh/t")]:
    m = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED)
    m.fit(Xtr, df3.loc[idx_tr, target])
    pred = m.predict(Xte)
    true = df3.loc[idx_te, target]
    models3[target] = m
    res3[target] = (true.values, pred)
    print(f"{target:7s}  R2 = {r2_score(true, pred):.3f}   MAE = {mean_absolute_error(true, pred):.4g} {unit}")

# ---- route recommendation --------------------------------------
def predict_routes(fe, si, cu):
    out = {}
    for r in ROUTES:
        row = pd.DataFrame([[fe, si, cu] + [int(r == x) for x in ROUTES]], columns=X3.columns)
        out[r] = tuple(models3[t].predict(row)[0] for t in ["Purity", "Cost", "Energy"])
    return out

def recommend(fe, si, cu):
    pr = predict_routes(fe, si, cu)
    ok = {r: v for r, v in pr.items() if v[0] >= REQUIRED_PURITY}
    if not ok:
        return None, pr
    return min(ok, key=lambda r: ok[r][1]), pr

def true_recommend(fe, si, cu):
    res = {r: simulate_stage3(fe, si, cu, r, noise=False) for r in ROUTES}
    ok = {r: v for r, v in res.items() if v[0] >= REQUIRED_PURITY}
    return min(ok, key=lambda r: ok[r][1]) if ok else None

print("\nExample: aluminium can, midpoint of Stage 2 ranges (Fe 0.50, Si 0.30, Cu 0.20)")
rec, pr = recommend(0.50, 0.30, 0.20)
for r, (p, c, e) in pr.items():
    print(f"  {r:16s} purity {p:7.3f} %   cost {c:8.0f} USD/t   energy {e:8.0f} kWh/t")
print("  Recommended route:", rec)

print("\nExample: clean aluminium foil (Fe 0.10, Si 0.08, Cu 0.03)")
rec2, pr2 = recommend(0.10, 0.08, 0.03)
for r, (p, c, e) in pr2.items():
    print(f"  {r:16s} purity {p:7.3f} %   cost {c:8.0f} USD/t   energy {e:8.0f} kWh/t")
print("  Recommended route:", rec2)

# recommendation over the test items (unique impurity profiles)
test_items = df3.loc[idx_te, ["Fe", "Si", "Cu"]].drop_duplicates()
recs, agree = [], 0
for fe, si, cu in test_items.values:
    r, _ = recommend(fe, si, cu)
    recs.append(r)
    agree += int(r == true_recommend(fe, si, cu))
counts = pd.Series(recs).fillna("None").value_counts()
print(f"\nRoute recommended for {len(test_items)} test items:")
for k, v in counts.items():
    print(f"  {k:16s} {v:3d}  ({100*v/len(test_items):.1f} %)")
print(f"Agreement with noise-free reference recommendation: {100*agree/len(test_items):.1f} %")


# ---- sensitivity to the required purity -------------------------
print("\nRecommended route for different purity requirements (100 test items):")
for thr in [99.0, 99.5, 99.9]:
    cnt = {"Remelting": 0, "Fluxing": 0, "Electrorefining": 0, "None": 0}
    for fe, si, cu in test_items.values:
        pr_ = predict_routes(fe, si, cu)
        ok = {r: v for r, v in pr_.items() if v[0] >= thr}
        cnt[min(ok, key=lambda r: ok[r][1]) if ok else "None"] += 1
    n = len(test_items)
    print(f"  required {thr:5.1f} %: " + ", ".join(f"{k} {100*v/n:.0f} %" for k, v in cnt.items()))

# ------------------------------------------------------------------
# STAGE 4 : simulated dataset  (400 records)
# ------------------------------------------------------------------
# ASSUMPTIONS (trends from Al-air literature, alkaline electrolyte):
#   - Fe, Cu and Si raise self-corrosion (Fe and Cu strongest)
#   - small Mg and Zn additions slightly reduce corrosion
#   - higher self-corrosion -> lower discharge (anode) efficiency
def make_stage4(n=400):
    rows = []
    for _ in range(n):
        total = rng.uniform(0.01, 1.00)                         # total impurity, wt%
        share = rng.dirichlet([3.0, 2.0, 1.5, 0.6, 0.6, 1.0])   # Fe Si Cu Mg Zn other
        fe, si, cu, mg, zn, oth = total * share
        purity = 100 - total
        corr = (0.18 + 2.6 * fe + 1.4 * cu + 0.9 * si - 0.5 * mg - 0.3 * zn)
        corr = max(0.05, corr * rng.normal(1, 0.06))            # mg cm-2 min-1
        eff = 88 - 22 * corr + rng.normal(0, 1.5)               # %
        eff = float(np.clip(eff, 20, 92))
        rows.append([purity, fe, si, cu, mg, zn, eff, corr])
    return pd.DataFrame(rows, columns=["Purity", "Fe", "Si", "Cu", "Mg", "Zn", "Efficiency", "Corrosion"])

df4 = make_stage4(400)
feat4 = ["Purity", "Fe", "Si", "Cu", "Mg", "Zn"]
X4tr, X4te, y4tr, y4te = train_test_split(df4[feat4], df4[["Efficiency", "Corrosion"]],
                                          test_size=0.2, random_state=SEED)
print("\n" + "=" * 60)
print("STAGE 4 : Battery performance prediction")
print("=" * 60)
print(f"Records: {len(df4)}   Train: {len(X4tr)}   Test: {len(X4te)}")
print(f"Purity range in data: {df4.Purity.min():.2f} - {df4.Purity.max():.2f} %")
models4, res4, imp4 = {}, {}, {}
for target, unit in [("Efficiency", "%"), ("Corrosion", "mg cm-2 min-1")]:
    m = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED)
    m.fit(X4tr, y4tr[target])
    pred = m.predict(X4te)
    models4[target] = m
    res4[target] = (y4te[target].values, pred)
    imp4[target] = pd.Series(m.feature_importances_, index=feat4).sort_values(ascending=False)
    print(f"{target:11s} R2 = {r2_score(y4te[target], pred):.3f}   MAE = {mean_absolute_error(y4te[target], pred):.4g} {unit}")
    print("  feature importance:", ", ".join(f"{k} {v:.3f}" for k, v in imp4[target].items()))

# ---- error carried forward from Stage 3 -------------------------
sd = np.sqrt(np.mean((res3["Purity"][0] - res3["Purity"][1]) ** 2))   # Stage 3 purity RMSE
Xn = X4te.copy()
Xn["Purity"] = Xn["Purity"] + rng.normal(0, sd, len(Xn))
print(f"\nError propagation: Stage 3 purity RMSE = {sd:.3f} %")
for target in ["Efficiency", "Corrosion"]:
    base = mean_absolute_error(y4te[target], models4[target].predict(X4te))
    noisy = mean_absolute_error(y4te[target], models4[target].predict(Xn))
    print(f"  {target:11s} MAE {base:.4g} -> {noisy:.4g} when purity carries Stage 3 error")

# ------------------------------------------------------------------
# FIGURES
# ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, t, u in zip(ax, ["Purity", "Cost"], ["Purity (%)", "Cost (USD/t)"]):
    tr, pr_ = res3[t]
    a.scatter(tr, pr_, s=12, alpha=0.7)
    lo, hi = min(tr.min(), pr_.min()), max(tr.max(), pr_.max())
    a.plot([lo, hi], [lo, hi], "r--", lw=1)
    a.set_xlabel("Actual " + u); a.set_ylabel("Predicted " + u)
    a.set_title(f"{t} (R$^2$ = {r2_score(tr, pr_):.3f})")
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/fig_stage3_pred_vs_actual.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, t in zip(ax, ["Efficiency", "Corrosion"]):
    s = imp4[t].sort_values()
    a.barh(s.index, s.values)
    a.set_title(f"Feature importance: {t}")
    a.set_xlabel("Importance")
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/fig_stage4_feature_importance.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, t, u in zip(ax, ["Efficiency", "Corrosion"], ["efficiency (%)", "corrosion (mg cm$^{-2}$ min$^{-1}$)"]):
    tr, pr_ = res4[t]
    a.scatter(tr, pr_, s=12, alpha=0.7)
    lo, hi = min(tr.min(), pr_.min()), max(tr.max(), pr_.max())
    a.plot([lo, hi], [lo, hi], "r--", lw=1)
    a.set_xlabel("Actual " + u); a.set_ylabel("Predicted " + u)
    a.set_title(f"{t} (R$^2$ = {r2_score(tr, pr_):.3f})")
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/fig_stage4_pred_vs_actual.png", dpi=200); plt.close()
print("\nFigures saved.")
