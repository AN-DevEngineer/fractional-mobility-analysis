# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 06_bilan_complet.py

# Une passe unique sur resultats_H/*_H.csv (19 150 segments).
# Calcule tous les chiffres nécessaires à la rédaction du chapitre 8.

# Sortie :
#   - resultats_H_1000/bilan_complet.txt   (console)
#   - resultats_H_1000/bilan_populations.csv (une ligne par population)
#   - resultats_H_1000/bootstrap_users.csv   (IC bootstrap par population)
# """

# import glob
# import os
# import numpy as np
# import pandas as pd

# SOURCE_GLOB   = "resultats_H/*_H.csv"
# SORTIE_DIR    = "resultats_H_1000"
# SEUIL_LONG    = 1000
# N_BOOTSTRAP   = 1000
# SEUIL_DIV     = 0.15

# COL_NUM = ["n_points", "h_dfa", "h_rs", "h_whittle", "divergence",
#            "dfa_bas", "dfa_haut", "h_rs_corrige", "nul_p", "pas_median_s"]

# os.makedirs(SORTIE_DIR, exist_ok=True)
# rng = np.random.default_rng(42)


# # ---------- 1. Chargement ----------
# fichiers = sorted(f for f in glob.glob(SOURCE_GLOB)
#                   if "synthese" not in f and "1000" not in f)
# print(f"Fichiers chargés : {len(fichiers)}")
# dfs = []
# for f in fichiers:
#     d = pd.read_csv(f)
#     d["user_id"] = os.path.basename(f).replace("_H.csv", "")
#     dfs.append(d)
# df = pd.concat(dfs, ignore_index=True)
# print(f"Segments totaux  : {len(df)}")
# print(f"Utilisateurs     : {df['user_id'].nunique()}")

# # Conversion numérique robuste
# for c in COL_NUM:
#     if c in df.columns:
#         df[c] = pd.to_numeric(df[c], errors="coerce")

# # Normalisation booléen whittle_a_la_borne
# if "whittle_a_la_borne" in df.columns:
#     df["whittle_a_la_borne"] = (
#         df["whittle_a_la_borne"].astype(str).str.strip().str.lower()
#         .isin(["true", "1", "yes", "vrai"])
#     )
# else:
#     df["whittle_a_la_borne"] = False


# # ---------- 2. Filtre gardé ----------
# garde = (
#     df["h_dfa"].notna() & df["h_rs"].notna() & df["h_whittle"].notna()
#     & (df["h_dfa"] > 0) & (df["h_dfa"] < 1)
#     & (df["h_rs"] > 0) & (df["h_rs"] < 1)
#     & (df["h_whittle"] > 0) & (df["h_whittle"] < 1)
#     & (~df["whittle_a_la_borne"])
#     & df["divergence"].notna() & (df["divergence"] <= SEUIL_DIV)
# )
# df["garde"] = garde
# df["ecart_dfa_rs"] = (df["h_dfa"] - df["h_rs"]).abs()


# # ---------- 3. Définition des populations ----------
# populations = {
#     "TOUS":               df,
#     "GARDES":             df[df["garde"]],
#     "GARDES_LONGS":       df[df["garde"] & (df["n_points"] >= SEUIL_LONG)],
#     "GARDES_COURTS":      df[df["garde"] & (df["n_points"] < SEUIL_LONG)],
#     "LONGS_NON_FILTRES":  df[df["n_points"] >= SEUIL_LONG],
#     "LONGS_TOUS_GARDES":  df[df["n_points"] >= SEUIL_LONG],  # alias
# }


# # ---------- 4. Fonction d'agrégation ----------
# def agreger(sub):
#     if len(sub) == 0:
#         return dict(n=0, n_users=0, h_dfa=np.nan, h_dfa_wavg=np.nan,
#                     h_rs=np.nan, h_wh=np.nan, div_moy=np.nan,
#                     ecart_dfa_rs_moy=np.nan, ecart_dfa_rs_med=np.nan,
#                     part_nul_sig=np.nan, dfa_bas_moy=np.nan,
#                     dfa_haut_moy=np.nan, ecart_pentes_moy=np.nan)
#     d = dict(
#         n=len(sub),
#         n_users=sub["user_id"].nunique(),
#         h_dfa=sub["h_dfa"].mean(),
#         h_rs=sub["h_rs"].mean(),
#         h_wh=sub["h_whittle"].mean(),
#         div_moy=sub["divergence"].mean(),
#         ecart_dfa_rs_moy=sub["ecart_dfa_rs"].mean(),
#         ecart_dfa_rs_med=sub["ecart_dfa_rs"].median(),
#     )
#     # Pondérée par n_points pour H_DFA
#     if sub["n_points"].sum() > 0:
#         d["h_dfa_wavg"] = (sub["h_dfa"] * sub["n_points"]).sum() / sub["n_points"].sum()
#     else:
#         d["h_dfa_wavg"] = np.nan
#     # nul_p : fraction significative à 0.05
#     if "nul_p" in sub.columns:
#         npv = sub["nul_p"].dropna()
#         d["part_nul_sig"] = (npv < 0.05).mean() if len(npv) else np.nan
#     else:
#         d["part_nul_sig"] = np.nan
#     # dfa_bas/haut/écart de pentes
#     for c in ["dfa_bas", "dfa_haut"]:
#         d[c + "_moy"] = sub[c].mean() if c in sub.columns else np.nan
#     if "dfa_bas" in sub.columns and "dfa_haut" in sub.columns:
#         ec = (sub["dfa_haut"] - sub["dfa_bas"]).dropna()
#         d["ecart_pentes_moy"] = ec.mean() if len(ec) else np.nan
#     else:
#         d["ecart_pentes_moy"] = np.nan
#     return d


# # ---------- 5. Tableau des populations ----------
# lignes = []
# print()
# print("=" * 100)
# print(f"{'POPULATION':<22} {'n':>6} {'users':>6} {'H_DFA':>8} {'H_DFAw':>8} "
#       f"{'H_RS':>8} {'H_WH':>8} {'div':>7} {'|D-R|':>7} {'nul<.05':>8}")
# print("=" * 100)

# for nom, sub in populations.items():
#     a = agreger(sub)
#     a["population"] = nom
#     lignes.append(a)
#     def f(x, fmt=".4f"): return f"{x:{fmt}}" if pd.notna(x) else "   —  "
#     print(f"{nom:<22} {a['n']:>6} {a['n_users']:>6} "
#           f"{f(a['h_dfa']):>8} {f(a['h_dfa_wavg']):>8} "
#           f"{f(a['h_rs']):>8} {f(a['h_wh']):>8} "
#           f"{f(a['div_moy']):>7} {f(a['ecart_dfa_rs_moy']):>7} "
#           f"{f(a['part_nul_sig'], '.2f'):>8}")
# print("=" * 100)


# # ---------- 6. Bootstrap par utilisateur ----------
# def bootstrap_user(sub, n_boot=N_BOOTSTRAP):
#     """Rééchantillonne les UTILISATEURS avec remise (les segments d'un user sont dépendants)."""
#     users = sub["user_id"].unique()
#     if len(users) < 2:
#         return dict(ic95_lo=np.nan, ic95_hi=np.nan, sd_boot=np.nan)
#     stats = []
#     for _ in range(n_boot):
#         tir = rng.choice(users, size=len(users), replace=True)
#         # Moyenne pondérée en concaténant les segments des users tirés
#         sous = sub[sub["user_id"].isin(tir)].copy()
#         # Répliquer chaque user selon le nombre de tirages
#         counts = pd.Series(tir).value_counts()
#         poids = sous["user_id"].map(counts).fillna(0)
#         wsum = (poids * sous["n_points"]).sum()
#         if wsum > 0:
#             stats.append(((sous["h_dfa"] * poids * sous["n_points"]).sum()) / wsum)
#     stats = np.array(stats)
#     return dict(
#         ic95_lo=float(np.percentile(stats, 2.5)),
#         ic95_hi=float(np.percentile(stats, 97.5)),
#         sd_boot=float(stats.std(ddof=1)),
#     )

# print()
# print("BOOTSTRAP PAR UTILISATEUR (B = 1000, pondéré par n_points, H_DFA)")
# print("=" * 100)
# boot_rows = []
# for nom in ["GARDES", "GARDES_LONGS", "GARDES_COURTS"]:
#     sub = populations[nom]
#     if len(sub) == 0:
#         continue
#     b = bootstrap_user(sub)
#     # Borne basse/haute sur la population observée
#     obs = agreger(sub)["h_dfa_wavg"]
#     print(f"{nom:<22} n={len(sub):>5} users={sub['user_id'].nunique():>3} "
#           f"H_pond={obs:.4f}  IC95=[{b['ic95_lo']:.4f}, {b['ic95_hi']:.4f}]  "
#           f"sd_boot={b['sd_boot']:.4f}")
#     boot_rows.append(dict(population=nom, n=len(sub),
#                           n_users=sub["user_id"].nunique(),
#                           h_dfa_wavg=obs, **b))
# print("=" * 100)


# # ---------- 7. dfa_bas / dfa_haut / écart de pentes sur GARDES ----------
# print()
# print("DIAGNOSTIC CROSSOVER (H_DFA) sur GARDES")
# print("=" * 100)
# g = df[df["garde"]]
# for c in ["dfa_bas", "dfa_haut"]:
#     if c in g.columns:
#         v = g[c].dropna()
#         print(f"  {c:<10}  n={len(v):>5}  moy={v.mean():.4f}  "
#               f"méd={v.median():.4f}  q25={v.quantile(.25):.4f}  q75={v.quantile(.75):.4f}")
# if "dfa_bas" in g.columns and "dfa_haut" in g.columns:
#     ec = (g["dfa_haut"] - g["dfa_bas"]).dropna()
#     print(f"  ecart      n={len(ec):>5}  moy={ec.mean():+.4f}  "
#           f"méd={ec.median():+.4f}")
#     print(f"  |ecart| > 0.10 : {100*(ec.abs() > 0.10).mean():.1f}% des gardés")
# print("=" * 100)


# # ---------- 8. Écriture CSV ----------
# chemin_pop = os.path.join(SORTIE_DIR, "bilan_populations.csv")
# pd.DataFrame(lignes).to_csv(chemin_pop, index=False)
# print(f"\n[OK] {chemin_pop}")

# chemin_boot = os.path.join(SORTIE_DIR, "bootstrap_users.csv")
# pd.DataFrame(boot_rows).to_csv(chemin_boot, index=False)
# print(f"[OK] {chemin_boot}")

"""Génère fig_dfa_exemple et fig_lois_exemple (figures d'illustration, pas des résultats)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(20261018)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

def fgn(n, H, rng):
    """fGn par plongement circulant (Davies-Harte)."""
    k = np.arange(0, n + 1)
    r = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    row = np.concatenate([r, r[-2:0:-1]])
    lam = np.fft.fft(row).real
    lam[lam < 0] = 0
    m = len(row)
    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    x = np.fft.fft(np.sqrt(lam / m) * z)
    return x.real[:n]

def dfa(x, scales, order=1):
    y = np.cumsum(x - x.mean())
    F = []
    for s in scales:
        ns = len(y) // s
        seg = y[: ns * s].reshape(ns, s)
        t = np.arange(s)
        res = []
        for v in seg:
            p = np.polyfit(t, v, order)
            res.append(np.mean((v - np.polyval(p, t)) ** 2))
        F.append(np.sqrt(np.mean(res)))
    return np.array(F)

# ---------- Figure 1 : principe de la DFA ----------
N = 4000
scales = np.unique(np.logspace(np.log10(4), np.log10(N // 4), 20).astype(int))
fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1]})
cols = {0.5: "tab:blue", 0.9: "tab:red"}
for i, H0 in enumerate([0.5, 0.9]):
    x = fgn(N, H0, rng)
    ax[0].plot(x + (2 - 5 * i) * 4, color=cols[H0], lw=0.6, label=("$H_0=%s$" % str(H0).replace(".", "{,}")))
    F = dfa(x, scales)
    sl = np.polyfit(np.log(scales), np.log(F), 1)
    ax[1].loglog(scales, F, "o", color=cols[H0], ms=4)
    ax[1].loglog(scales, np.exp(np.polyval(sl, np.log(scales))), "-", color=cols[H0],
                 label=("$H_0=%s$ : pente $\\hat H=%s$" % (str(H0).replace(".", "{,}"), ("%.2f" % sl[0]).replace(".", "{,}"))))
ax[0].set_title("Séries simulées (fGn)"); ax[0].set_xlabel("temps $t$"); ax[0].set_yticks([])
ax[0].legend(loc="upper right", fontsize=8)
ax[1].set_title("Fonction de fluctuation $F(s)$"); ax[1].set_xlabel("échelle $s$"); ax[1].set_ylabel("$F(s)$")
ax[1].legend(fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig_dfa_exemple.{ext}", dpi=200)
plt.close(fig)

# ---------- Figure 2 : lois pure / tronquée / log-normale ----------
x = np.logspace(0, 3, 600)
a, lam = 2.4, 0.01
sf_pure = x ** (1 - a)
grid = np.logspace(0, 6, 200000)
pdf = grid ** (-a) * np.exp(-lam * grid)
cum = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
sf_tr = np.interp(x, grid, (cum[-1] - cum) / cum[-1])
ln = stats.lognorm(s=1.1, scale=np.exp(0.3))
sf_ln = ln.sf(x) / ln.sf(1.0)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.loglog(x, sf_pure, color="tab:blue", label=r"loi de puissance pure ($\alpha=2{,}4$)")
ax.loglog(x, sf_tr, color="tab:red", label=r"loi tronquée ($\lambda=0{,}01$)")
ax.loglog(x, sf_ln, color="tab:green", ls="--", label="log-normale")
ax.set_ylim(1e-5, 1.5); ax.set_xlabel("$x$"); ax.set_ylabel("$P(X>x)$"); ax.legend(fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig_lois_exemple.{ext}", dpi=200)