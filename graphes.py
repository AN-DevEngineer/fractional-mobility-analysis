# #!/usr/bin/env python3
# """
# Génère les graphes du mémoire : 8 groupes sur H/dataset, 5 sur les lois de puissance.
# Sortie : memoire/figures/graphe_*.pdf

# Structure H/dataset :
#   1a, 1b, 1c, 1d : validation fGn par méthode (DFA, R/S brut, R/S corrigé, Whittle)
#   2              : distribution H unifiée (3 méthodes, axe y tronqué à 10%)
#   3a, 3b, 3c     : scatters par paires (DFA/R/S, DFA/Whittle, R/S/Whittle)
#   4              : H par user + IC bootstrap
#   5              : heatmap calibration
#   6              : audit T-Drive
#   7              : distribution longueurs segments
#   8              : distribution Δt

# Structure LP :
#   9 à 13.

# Usage : python graphes.py
# """

# import glob
# import json
# import os
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from scipy import integrate

# from lois_de_puissance.alternatives import _log_normalisation_puissance_tronquee
# from pretraitement.arrets import distances_entre_arrets


# def sauvegarder(fig, chemin_pdf):
#     """Sauvegarde le graphe en PDF et PNG (même chemin, extension différente)."""
#     fig.savefig(chemin_pdf)
#     fig.savefig(chemin_pdf.replace(".pdf", ".png"))


# plt.rcParams.update({
#     "figure.dpi": 150, "savefig.dpi": 300,
#     "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
#     "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
#     "savefig.bbox": "tight", "savefig.format": "pdf",
#     "axes.grid": True, "grid.alpha": 0.3,
# })

# FIG_DIR = "memoire/figures"
# os.makedirs(FIG_DIR, exist_ok=True)

# COUL_GARDE = "C0"
# COUL_EXCLU = "grey"
# COUL_REF = "C3"
# COUL_SATUR = "darkorange"

# # Seuils constants
# WHITTLE_SATURATION = 0.999
# SEUIL_FORT = 0.5

# # Fichiers LP (produits par 04_lois_de_puissance.py)
# LP_CSV_AGREGE = "resultats_H/lois_de_puissance/lois_resume_rapide_avec_alt_B1000.csv"
# LP_CSV_SEJOURS = "resultats_H/lois_de_puissance/sejour_rapide_avec_alt_B1000.csv"
# LP_CSV_4JEUX = "resultats_H/lois_de_puissance/lois_resume.csv"


# # ── Helpers ─────────────────────────────────────────────────────────────────
# def _parse_bool(x):
#     if x is None:
#         return None
#     if isinstance(x, (bool, np.bool_)):
#         return bool(x)
#     try:
#         if pd.isna(x):
#             return None
#     except (TypeError, ValueError):
#         pass
#     s = str(x).strip().lower()
#     if s in ("", "nan"):
#         return None
#     return s in ("true", "1", "yes", "vrai", "o", "v")


# def ajouter_garde(df):
#     for c in ("h_dfa", "h_rs", "h_whittle", "divergence"):
#         df[c] = pd.to_numeric(df[c], errors="coerce")
#     wh = df["whittle_a_la_borne"].apply(_parse_bool)
#     df["garde"] = (
#         df["h_dfa"].notna() & df["h_rs"].notna() & df["h_whittle"].notna()
#         & (df["h_dfa"] > 0) & (df["h_dfa"] < 1)
#         & (df["h_rs"] > 0) & (df["h_rs"] < 1)
#         & (df["h_whittle"] > 0) & (df["h_whittle"] < 1)
#         & (wh == False)
#         & df["divergence"].notna() & (df["divergence"] <= 0.15)
#     )
#     return df


# def charger_tous_segments():
#     fichiers = sorted(glob.glob("resultats_H/*_H.csv"))
#     fichiers = [f for f in fichiers
#                 if "synthese" not in f and "morceaux" not in f]
#     print(f"Chargement de {len(fichiers)} fichiers per-segment...")
#     dfs = []
#     for f in fichiers:
#         try:
#             dfs.append(pd.read_csv(f))
#         except Exception as e:
#             print(f"  [WARN] {f} : {e}")
#     df = ajouter_garde(pd.concat(dfs, ignore_index=True))
#     print(f"  Total segments : {len(df)}")
#     print(f"  Segments gardés : {int(df['garde'].sum())}")
#     if len(df) != 19150:
#         print(f"  [ATTENTION] Attendu 19150, obtenu {len(df)}")
#     if int(df["garde"].sum()) != 7853:
#         print(f"  [ATTENTION] Attendu 7853 gardés, obtenu {int(df['garde'].sum())}")
#     return df


# # ── Helpers LP ──────────────────────────────────────────────────────────────
# def _ccdf_empirique(x):
#     """CCDF empirique : P(X >= x_i) pour x_i triés croissants."""
#     x = np.sort(np.asarray(x, dtype=float))
#     n = len(x)
#     return x, (n - np.arange(n)) / n


# def _ccdf_lp(x, alpha, xmin):
#     """CCDF de la loi de puissance pure, conditionnelle à x >= xmin."""
#     return (x / xmin) ** (1.0 - alpha)


# def _ccdf_tronquee(x, alpha, lam, xmin):
#     """CCDF de la tronquée conditionnelle."""
#     log_Z = _log_normalisation_puissance_tronquee(alpha, lam, xmin)
#     if not np.isfinite(log_Z):
#         return np.full_like(x, np.nan)
#     Z = np.exp(log_Z)
#     out = np.empty(len(x))
#     for i, xi in enumerate(x):
#         if xi < xmin:
#             out[i] = np.nan
#             continue
#         val, _ = integrate.quad(lambda t: t ** (-alpha) * np.exp(-lam * t),
#                                 xmin, xi, limit=100)
#         out[i] = 1.0 - val / Z
#     return out


# def _parametres_lp(variable, csv=LP_CSV_AGREGE):
#     """Paramètres (xmin, alpha, alpha_tronq, lambda) pour la ligne agrege d'une variable."""
#     df = pd.read_csv(csv)
#     ligne = df[(df["population"] == "agrege") & (df["variable"] == variable)].iloc[0]
#     params = json.loads(ligne["parametres_alternatives"]) if pd.notna(ligne["parametres_alternatives"]) else {}
#     tronq = params.get("puissance_tronquee", {})
#     return dict(
#         xmin=float(ligne["xmin"]),
#         alpha=float(ligne["alpha"]),
#         alpha_tronq=float(tronq["alpha"]) if "alpha" in tronq else np.nan,
#         lambda_tronq=float(tronq["lambda"]) if "lambda" in tronq else np.nan,
#         n_queue=int(ligne["n_queue"]),
#     )


# def _filtre_sejour(df):
#     """Durées de séjour non censurées (interne uniquement), comme le pipeline LP."""
#     d = df.copy()
#     d["censuree"] = d["censuree"].astype(str).str.lower()
#     mask = (d["type"] == "interne") & (d["censuree"] == "false")
#     return d.loc[mask, "duree_s"].to_numpy(dtype=float)


# # ═══════════════════════════════════════════════════════════════════════════
# # Graphes 1a–1d : validation fGn, un fichier par méthode
# # ═══════════════════════════════════════════════════════════════════════════

# def _graphe_validation_methode(methode_code, titre, filename, annoter_biais=False):
#     path = "resultats_H/validation/exactitude_resume.csv"
#     if not os.path.exists(path):
#         print(f"[SKIP {filename}] {path} introuvable")
#         return
#     df = pd.read_csv(path)
#     sub = df[df["methode"] == methode_code]
#     if sub.empty:
#         print(f"[SKIP {filename}] méthode {methode_code} absente")
#         return

#     fig, ax = plt.subplots(figsize=(7, 5))
#     for n in sorted(sub["n"].unique()):
#         s = sub[sub["n"] == n].sort_values("h0")
#         ax.errorbar(s["h0"], s["h0"] + s["biais"], yerr=s["rmse"],
#                     marker="o", capsize=3, label=f"N={n}")

#     x = np.array([0.05, 0.95])
#     ax.fill_between(x, x - 0.05, x + 0.05, alpha=0.15, color="green",
#                     label="Biais acceptable $\\pm 0{,}05$")
#     ax.plot(x, x, "k--", alpha=0.5, label="y = x")
#     ax.axhline(0.5, color=COUL_REF, linestyle=":", lw=1)
#     ax.set_xlim(0.05, 0.95)
#     ax.set_ylim(0.05, 1.15)
#     ax.set_xlabel("$H_0$ (valeur vraie)")
#     ax.set_ylabel("$\\hat H$ (estimation)")
#     ax.set_title(titre)
#     ax.legend(loc="upper left", fontsize=8)

#     if annoter_biais and (sub["n"].min() in sub["n"].values):
#         n_min = sub["n"].min()
#         s09 = sub[(sub["n"] == n_min) & (sub["h0"] == sub["h0"].max())]
#         if not s09.empty:
#             b = float(s09["biais"].iloc[0])
#             h0 = float(s09["h0"].iloc[0])
#             ax.annotate(f"biais = {b:+.3f}",
#                         xy=(h0, h0 + b),
#                         xytext=(0.7, 0.55), fontsize=9, color=COUL_REF,
#                         arrowprops=dict(arrowstyle="->", color=COUL_REF))

#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_{filename}.pdf")
#     plt.close(fig)
#     print(f"[OK {filename}] {titre}")


# def graphe_1a_validation_dfa():
#     _graphe_validation_methode("dfa", "Validation DFA sur fGn de $H_0$ connu",
#                                 "1a_validation_dfa")


# def graphe_1b_validation_rs_brut():
#     _graphe_validation_methode("rs_brut", "Validation R/S brut sur fGn de $H_0$ connu",
#                                 "1b_validation_rs_brut")


# def graphe_1c_validation_rs_corrige():
#     _graphe_validation_methode("rs_corrige", "Validation R/S corrigé sur fGn de $H_0$ connu",
#                                 "1c_validation_rs_corrige", annoter_biais=True)


# def graphe_1d_validation_whittle():
#     _graphe_validation_methode("whittle", "Validation Whittle local sur fGn de $H_0$ connu",
#                                 "1d_validation_whittle")


# # ═══════════════════════════════════════════════════════════════════════════
# # Graphe 2 : distribution H unifiée (option a)
# # ═══════════════════════════════════════════════════════════════════════════

# def graphe_2_distribution(df):
#     cols = [("h_dfa", "$\\hat H_{\\mathrm{DFA}}$", "C0"),
#             ("h_rs", "$\\hat H_{\\mathrm{R/S}}$", "C1"),
#             ("h_whittle", "$\\hat H_{\\mathrm{Whittle}}$", "C2")]
#     bins = np.linspace(0, 1.5, 75)

#     fig, ax = plt.subplots(figsize=(9, 5))
#     for col, label, color in cols:
#         tous = df[col].dropna()
#         weights = np.ones(len(tous)) / len(tous) * 100   # en %
#         ax.hist(tous, bins=bins, weights=weights, alpha=0.5,
#                 color=color, edgecolor="white", linewidth=0.4,
#                 label=f"{label} (n={len(tous):,})")

#     # Annotation du pic Whittle (tronqué visuellement)
#     n_sat = int((df["h_whittle"] >= WHITTLE_SATURATION - 1e-6).sum())
#     n_valid_wh = int(df["h_whittle"].notna().sum())
#     pct_sat = 100 * n_sat / n_valid_wh if n_valid_wh else 0
#     ax.annotate(f"Whittle saturé : {pct_sat:.0f} % à $\\hat H = 0.999$\n"
#                 f"({n_sat:,} segments, tronqué visuellement)",
#                 xy=(0.999, 10), xytext=(0.72, 8),
#                 fontsize=9, color=COUL_SATUR,
#                 arrowprops=dict(arrowstyle="->", color=COUL_SATUR, lw=1.2),
#                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
#                           edgecolor=COUL_SATUR, alpha=0.9))

#     ax.set_ylim(0, 10)
#     ax.set_xlim(0, 1.5)
#     ax.set_xlabel("$\\hat H$ (valeur estimée, tous segments)")
#     ax.set_ylabel("Pourcentage de segments par bin (%)")
#     ax.axvline(0.5, color=COUL_REF, linestyle=":", lw=1, alpha=0.6)
#     ax.legend(loc="upper left", fontsize=9)
#     ax.set_title("Distribution de $\\hat H$ pour les trois méthodes (tous segments)")
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_2_distribution.pdf")
#     plt.close(fig)
#     print("[OK 2] distribution unifiée")


# # ═══════════════════════════════════════════════════════════════════════════
# # Graphes 3a–3c : scatters par paires (Version A)
# # ═══════════════════════════════════════════════════════════════════════════

# def _scatter_paire(df, col_x, col_y, label_x, label_y, filename,
#                     xlim_max=1.55, ylim_max=1.55, ligne_saturation_x=False):
#     sub = df.dropna(subset=[col_x, col_y])
#     exclus = sub[~sub["garde"]]
#     gardes = sub[sub["garde"]]

#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.scatter(exclus[col_x], exclus[col_y], s=3, alpha=0.15,
#                color=COUL_EXCLU, label=f"Exclus (n={len(exclus):,})",
#                rasterized=True, edgecolors="none")
#     ax.scatter(gardes[col_x], gardes[col_y], s=3, alpha=0.6,
#                color=COUL_GARDE, label=f"Gardés (n={len(gardes):,})",
#                rasterized=True, edgecolors="none")

#     x = np.array([-0.05, max(xlim_max, ylim_max)])
#     ax.plot(x, x, "k-", lw=0.6, alpha=0.6, label="y = x")
#     ax.axhline(SEUIL_FORT, color=COUL_REF, linestyle=":", lw=1, alpha=0.6)
#     ax.axvline(SEUIL_FORT, color=COUL_REF, linestyle=":", lw=1, alpha=0.6)

#     # Ligne de saturation Whittle si applicable
#     if ligne_saturation_x:
#         n_sat = int((sub[col_x] >= WHITTLE_SATURATION - 1e-6).sum())
#         ax.axvline(WHITTLE_SATURATION, color=COUL_SATUR, linestyle="--",
#                    lw=1.2, alpha=0.8,
#                    label=f"Whittle saturé (0.999, n={n_sat:,})")

#     # Annotation : combien de gardés ont les deux H > 0.5
#     n_forts = int((gardes[[col_x, col_y]].min(axis=1) > SEUIL_FORT).sum())
#     texte = (f"{n_forts:,} / {len(gardes):,} gardés\n"
#              f"ont les deux méthodes $> 0{{,}}5$")
#     if ligne_saturation_x:
#         texte += ("\n\n$\\hat H = 0{,}999$ : borne sup.\n"
#                   "de la grille Whittle\n(non-stationnarité)")
#     ax.text(0.04, 0.96, texte,
#             transform=ax.transAxes, fontsize=9,
#             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
#                       edgecolor=COUL_SATUR if ligne_saturation_x else COUL_GARDE,
#                       alpha=0.9),
#             verticalalignment="top")

#     ax.set_xlim(-0.05, xlim_max)
#     ax.set_ylim(-0.05, ylim_max)
#     ax.set_xlabel(label_x)
#     ax.set_ylabel(label_y)
#     ax.legend(loc="lower right", markerscale=3, fontsize=8)
#     if xlim_max == ylim_max:
#         ax.set_aspect("equal")
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_{filename}.pdf")
#     plt.close(fig)
#     print(f"[OK {filename}] {label_x} vs {label_y}")


# def graphe_3a_dfa_rs(df):
#     _scatter_paire(df, "h_rs", "h_dfa",
#                     "$\\hat H_{\\mathrm{R/S}}$ (brut)", "$\\hat H_{\\mathrm{DFA}}$",
#                     "3a_dfa_rs")


# def graphe_3b_dfa_wh(df):
#     _scatter_paire(df, "h_whittle", "h_dfa",
#                     "$\\hat H_{\\mathrm{Whittle}}$", "$\\hat H_{\\mathrm{DFA}}$",
#                     "3b_dfa_wh",
#                     xlim_max=1.05, ylim_max=1.55,
#                     ligne_saturation_x=True)


# def graphe_3c_rs_wh(df):
#     _scatter_paire(df, "h_whittle", "h_rs",
#                     "$\\hat H_{\\mathrm{Whittle}}$", "$\\hat H_{\\mathrm{R/S}}$",
#                     "3c_rs_wh",
#                     xlim_max=1.05, ylim_max=1.55,
#                     ligne_saturation_x=True)


# # ── Graphe 4 : H par user + IC bootstrap ────────────────────────────────────
# def graphe_4_user_ic():
#     fichiers = sorted(glob.glob("resultats_H/*_H.csv"))
#     fichiers = [f for f in fichiers
#                 if "synthese" not in f and "morceaux" not in f]
#     rng = np.random.default_rng(42)
#     resultats = []
#     for f in fichiers:
#         user = os.path.basename(f).replace("_H.csv", "")
#         try:
#             df_u = ajouter_garde(pd.read_csv(f))
#         except Exception:
#             continue
#         vals = df_u.loc[df_u["garde"], "h_dfa"].dropna().values
#         if len(vals) < 2:
#             continue
#         boots = np.array([np.median(rng.choice(vals, len(vals), replace=True))
#                           for _ in range(1000)])
#         lo, hi = np.percentile(boots, [2.5, 97.5])
#         resultats.append(dict(user=user, n=len(vals),
#                               h=float(np.median(vals)), lo=lo, hi=hi))
#     if not resultats:
#         print("[SKIP 4]")
#         return
#     res = pd.DataFrame(resultats).sort_values("h").reset_index(drop=True)
#     fig, ax = plt.subplots(figsize=(8, 7))
#     y = np.arange(len(res))
#     ax.hlines(y, res["lo"], res["hi"], colors=COUL_GARDE, alpha=0.35, lw=1)
#     sc = ax.scatter(res["h"], y, c=res["n"], cmap="viridis",
#                     s=12, zorder=3, edgecolors="none")
#     plt.colorbar(sc, ax=ax, label="$n_{\\mathrm{gardés}}$")
#     i_max = res["n"].idxmax(); i_min = res["n"].idxmin()
#     for i, prefix in [(i_max, "plus gros"), (i_min, "plus petit")]:
#         ax.annotate(f"{prefix} : user {res.loc[i, 'user']}\n"
#                     f"(n={res.loc[i, 'n']})",
#                     xy=(res.loc[i, "h"], i),
#                     xytext=(res.loc[i, "h"] + 0.05, i + 20),
#                     fontsize=8, alpha=0.8,
#                     arrowprops=dict(arrowstyle="->", alpha=0.5, lw=0.7))
#     ax.axvline(0.5, color=COUL_REF, linestyle=":", lw=1, label="H = 0.5")
#     ax.axvline(0.749, color="k", linestyle="--", lw=1,
#                label="Moyenne pondérée = 0.749")
#     ax.set_yticks([])
#     ax.set_xlabel("$\\hat H_{\\mathrm{DFA}}$ médian par utilisateur "
#                   "(IC 95 % bootstrap)")
#     ax.set_ylabel(f"{len(res)} utilisateurs triés par $\\hat H$")
#     ax.legend(loc="lower right", fontsize=8)
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_4_user_ic.pdf")
#     plt.close(fig)
#     print("[OK 4] user_ic")


# # ── Graphe 5 : Heatmap calibration ──────────────────────────────────────────
# def graphe_5_heatmap():
#     source = None
#     for pattern in ("resultats_grille/*.csv", "resultats_grille/*/*.csv"):
#         files = sorted(glob.glob(pattern))
#         if files:
#             source = files[0]
#             break
#     gaps = [300, 450, 600, 900, 1200, 1800]
#     min_pts = [100, 200, 300, 400, 500]
#     data = None
#     if source:
#         try:
#             df = pd.read_csv(source)
#             gcol = next((c for c in ("gap", "TIME_GAP", "seuil_gap", "threshold")
#                          if c in df.columns), None)
#             mcol = next((c for c in ("min_points", "MIN_POINTS", "n_min")
#                          if c in df.columns), None)
#             tcol = next((c for c in ("taux", "taux_conservation", "pct", "conservation")
#                          if c in df.columns), None)
#             if gcol and mcol and tcol:
#                 piv = df.pivot_table(index=mcol, columns=gcol, values=tcol)
#                 gaps, min_pts = list(piv.columns), list(piv.index)
#                 data = piv.values
#                 print(f"[5] Chargé depuis {source}")
#         except Exception as e:
#             print(f"[WARN 5] Échec lecture {source} : {e}")
#     if data is None:
#         print("[5] Utilisation des valeurs du rapport §3 (tab:calibration)")
#         data = np.array([
#             [97.3, 98.0, 98.3, 98.6, 98.8, 99.0],
#             [91.2, 93.5, 94.5, 95.5, 96.0, 96.5],
#             [84.4, 87.9, 89.6, 91.3, 92.2, 93.1],
#             [77.5, 81.9, 84.2, 86.7, 87.9, 89.3],
#             [71.2, 76.2, 79.1, 82.1, 83.7, 85.4],
#         ])
#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     im = ax.imshow(data, cmap="viridis", aspect="auto")
#     for i in range(len(min_pts)):
#         for j in range(len(gaps)):
#             v = data[i, j]
#             col = "white" if v < (data.min() + data.max()) / 2 else "black"
#             ax.text(j, i, f"{v:.1f}", ha="center", va="center",
#                     color=col, fontsize=9)
#     try:
#         j900 = gaps.index(900); i300 = min_pts.index(300)
#         ax.add_patch(plt.Rectangle((j900 - 0.5, i300 - 0.5), 1, 1,
#                                    fill=False, edgecolor=COUL_REF, lw=2.5))
#     except ValueError:
#         pass
#     ax.set_xticks(range(len(gaps))); ax.set_xticklabels(gaps)
#     ax.set_yticks(range(len(min_pts))); ax.set_yticklabels(min_pts)
#     ax.set_xlabel("$T_{\\mathrm{gap}}$ (s)")
#     ax.set_ylabel("$N_{\\min}$ (points)")
#     ax.set_title("Taux de conservation des points (%)")
#     plt.colorbar(im, ax=ax, label="% des points conservés")
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_5_heatmap.pdf")
#     plt.close(fig)
#     print("[OK 5] heatmap")


# # ── Graphe 6 : Audit T-Drive ────────────────────────────────────────────────
# def graphe_6_tdrive():
#     labels = [
#         "Taxis avec $v_{\\max} > 200$ km/h",
#         "Taxis avec $v_{\\max} > 1000$ km/h",
#         "Fichiers avec coordonnées $(0,0)$",
#         "Fichiers avec $< 300$ points valides",
#     ]
#     valeurs = [65.3, 48.6, 4.2, 9.1]
#     fig, ax = plt.subplots(figsize=(8, 3.5))
#     y = np.arange(len(labels))
#     colors = [COUL_REF, COUL_REF, "grey", "grey"]
#     ax.barh(y, valeurs, color=colors)
#     for i, v in enumerate(valeurs):
#         ax.text(v + 1, i, f"{v:.1f} %", va="center", fontsize=9)
#     ax.set_yticks(y); ax.set_yticklabels(labels)
#     ax.set_xlim(0, 80)
#     ax.set_xlabel("Pourcentage du corpus T-Drive")
#     ax.set_title("Audit structurel de T-Drive : anomalies détectées")
#     ax.invert_yaxis()
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_6_tdrive.pdf")
#     plt.close(fig)
#     print("[OK 6] tdrive")


# # ── Graphe 7 : Distribution longueurs segments ──────────────────────────────
# def graphe_7_longueurs(df):
#     n = df["n_points"].dropna().values
#     fig, ax = plt.subplots(figsize=(7, 4))
#     bins = np.logspace(np.log10(max(n.min(), 10)), np.log10(n.max()), 50)
#     ax.hist(n, bins=bins, alpha=0.75, color=COUL_GARDE,
#             edgecolor="white", linewidth=0.5)
#     ax.axvline(300, color=COUL_REF, linestyle="--", lw=1.5,
#                label=f"$N_{{\\min}} = 300$ (seuil)")
#     ax.axvline(np.median(n), color="k", linestyle=":", lw=1.5,
#                label=f"Médiane = {int(np.median(n))}")
#     pct_sous_1000 = 100 * (n < 1000).mean()
#     ax.set_xscale("log")
#     ax.set_xlabel("Nombre de points par segment (log)")
#     ax.set_ylabel("Nombre de segments")
#     ax.set_title(f"Longueurs de segments : médiane {int(np.median(n))} "
#                  f"points ({pct_sous_1000:.0f} % < 1\\,000)")
#     ax.legend(fontsize=8)
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_7_longueurs.pdf")
#     plt.close(fig)
#     print("[OK 7] longueurs")


# # ── Graphe 8 : Distribution Δt (log-log) ────────────────────────────────────
# def graphe_8_deltas():
#     fichiers = glob.glob("sortie/*.csv")
#     if not fichiers:
#         print("[SKIP 8]")
#         return
#     tous_dt = []
#     for f in fichiers[:30]:
#         try:
#             df = pd.read_csv(f, usecols=["timestamp", "segment_id"])
#         except Exception:
#             continue
#         for _, g in df.groupby("segment_id", sort=False):
#             t = g["timestamp"].to_numpy()
#             if len(t) < 2:
#                 continue
#             dt = np.diff(t); dt = dt[dt > 0]
#             if len(dt):
#                 tous_dt.append(dt)
#     if not tous_dt:
#         print("[SKIP 8]")
#         return
#     dt = np.concatenate(tous_dt)
#     fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.5),
#                                       gridspec_kw={"width_ratios": [1, 1.2]})
#     bins_all = np.logspace(np.log10(max(dt.min(), 0.5)),
#                            np.log10(dt.max()), 80)
#     ax.hist(dt, bins=bins_all, alpha=0.8, color=COUL_GARDE)
#     ax.axvline(900, color=COUL_REF, linestyle="--", lw=1.5,
#                label="$T_{\\mathrm{gap}} = 900$ s")
#     ax.set_xscale("log"); ax.set_yscale("log")
#     ax.set_xlabel("$\\Delta t$ (s, log)")
#     ax.set_ylabel("Occurrences (log)")
#     ax.set_title("Vue globale (0.5–10$^3$ s)")
#     ax.legend(fontsize=8)
#     dt_zoom = dt[(dt >= 10) & (dt <= 1000)]
#     bins_zoom = np.logspace(1, 3, 60)
#     ax_zoom.hist(dt_zoom, bins=bins_zoom, alpha=0.8, color=COUL_GARDE)
#     ax_zoom.axvline(900, color=COUL_REF, linestyle="--", lw=1.5,
#                     label="$T_{\\mathrm{gap}} = 900$ s")
#     ax_zoom.set_xscale("log"); ax_zoom.set_yscale("log")
#     ax_zoom.set_xlabel("$\\Delta t$ (s, log)")
#     ax_zoom.set_ylabel("Occurrences (log)")
#     ax_zoom.set_title("Zoom [10, 1000] : décroissance monotone,\naucun mode secondaire")
#     ax_zoom.legend(fontsize=8)
#     fig.suptitle("Distribution des $\\Delta t$ sur GeoLife")
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_8_deltas.pdf")
#     plt.close(fig)
#     print("[OK 8] deltas")


# # ═══════════════════════════════════════════════════════════════════════════
# # Lois de puissance — graphes 9 à 13 (inchangés)
# # ═══════════════════════════════════════════════════════════════════════════

# def graphe_9_ccdf_sejour():
#     if not (os.path.exists(LP_CSV_SEJOURS) and os.path.exists(LP_CSV_AGREGE)):
#         print("[SKIP 9] CSV LP introuvables")
#         return
#     durees = _filtre_sejour(pd.read_csv(LP_CSV_SEJOURS))
#     p = _parametres_lp("sejour")
#     durees = durees[durees >= p["xmin"]]
#     x_emp, ccdf_emp = _ccdf_empirique(durees)
#     x_theo = np.logspace(np.log10(p["xmin"]), np.log10(x_emp.max()), 300)
#     ccdf_lp = _ccdf_lp(x_theo, p["alpha"], p["xmin"])
#     ccdf_tr = _ccdf_tronquee(x_theo, p["alpha_tronq"], p["lambda_tronq"], p["xmin"])
#     fig, ax = plt.subplots(figsize=(8, 5.5))
#     ax.plot(x_emp, ccdf_emp, "o", ms=2.5, alpha=0.4, color=COUL_EXCLU,
#             label=f"Observé (n={len(durees)})")
#     ax.plot(x_theo, ccdf_lp, "-", lw=2, color=COUL_GARDE,
#             label=f"LP pure ($\\alpha={p['alpha']:.2f}$)")
#     ax.plot(x_theo, ccdf_tr, "-", lw=2, color=COUL_REF,
#             label=f"Tronquée ($\\alpha={p['alpha_tronq']:.2f}$, "
#                   f"$\\lambda={p['lambda_tronq']:.1e}$)")
#     ax.axvline(p["xmin"], ls=":", color="k", alpha=0.5)
#     ax.text(p["xmin"] * 1.05, ax.get_ylim()[1] * 0.6, " $x_{\\min}$", fontsize=9)
#     if p["lambda_tronq"] > 0:
#         echelle = 1.0 / p["lambda_tronq"]
#         ax.axvline(echelle, ls="--", color=COUL_REF, alpha=0.5)
#         ax.text(echelle * 0.95, ax.get_ylim()[1] * 0.25,
#                 f" $1/\\lambda\\approx{echelle/3600:.1f}$ h",
#                 fontsize=9, color=COUL_REF, ha="right")
#     ax.set_xscale("log"); ax.set_yscale("log")
#     ax.set_xlabel("Durée de séjour (s)")
#     ax.set_ylabel("$P(X \\geq x)$")
#     ax.set_title("CCDF des durées de séjour (agrégé, $x \\geq x_{\\min}$)")
#     ax.legend(loc="lower left", fontsize=9)
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_9_ccdf_sejour.pdf")
#     plt.close(fig)
#     print("[OK 9] ccdf_sejour")


# def graphe_10_aicc():
#     if not os.path.exists(LP_CSV_AGREGE):
#         print("[SKIP 10]")
#         return
#     df = pd.read_csv(LP_CSV_AGREGE)
#     agrege = df[df["population"] == "agrege"].copy()
#     ordre_var = ["sejour", "distance", "sejour_avec_censurees", "distance_a_cheval"]
#     labels = ["séjour", "distance\n(même seg.)", "séjour\n(censurés)", "distance\n(à cheval)"]
#     agrege = agrege.set_index("variable").loc[ordre_var]
#     modeles = ["loi_de_puissance", "exponentielle", "lognormale", "weibull", "puissance_tronquee"]
#     noms = ["LP", "Exp", "Log-norm", "Weibull", "Tronquée"]
#     couleurs = [COUL_GARDE, "C1", "C2", "C4", COUL_REF]
#     fig, ax = plt.subplots(figsize=(9, 5))
#     bas = np.zeros(len(agrege))
#     for m, nom, c in zip(modeles, noms, couleurs):
#         vals = agrege[f"poids_aicc_{m}"].to_numpy(dtype=float)
#         ax.bar(labels, vals, bottom=bas, label=nom, color=c,
#                edgecolor="white", lw=0.5)
#         bas += vals
#     ax.set_ylabel("Poids AICc")
#     ax.set_ylim(0, 1.02)
#     ax.set_title("Poids AICc des 5 modèles, par variable (agrégé)")
#     ax.legend(loc="upper right", ncol=5, fontsize=9)
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_10_aicc.pdf")
#     plt.close(fig)
#     print("[OK 10] aicc")


# def graphe_11_pvalues():
#     if not os.path.exists(LP_CSV_AGREGE):
#         print("[SKIP 11]")
#         return
#     df = pd.read_csv(LP_CSV_AGREGE)
#     agrege = df[df["population"] == "agrege"].copy()
#     ordre_var = ["sejour", "distance", "sejour_avec_censurees", "distance_a_cheval"]
#     labels = ["séjour", "distance\n(même seg.)", "séjour\n(censurés)", "distance\n(à cheval)"]
#     agrege = agrege.set_index("variable").loc[ordre_var]
#     x = np.arange(len(agrege))
#     largeur = 0.38
#     p_lp = agrege["p_bootstrap"].to_numpy(dtype=float)
#     p_tr = agrege["p_adequation_tronquee"].to_numpy(dtype=float)
#     fig, ax = plt.subplots(figsize=(9, 5))
#     b1 = ax.bar(x - largeur / 2, p_lp, largeur, label="LP pure (Clauset)",
#                 color=["C2" if p > 0.1 else COUL_REF for p in p_lp])
#     b2 = ax.bar(x + largeur / 2, p_tr, largeur, label="Tronquée (bootstrap)",
#                 color=["C2" if p > 0.1 else COUL_REF for p in p_tr], alpha=0.7)
#     ax.axhline(0.1, ls="--", color="k", label="seuil 0.1")
#     ax.set_xticks(x); ax.set_xticklabels(labels)
#     ax.set_ylabel("p-valeur")
#     ax.set_ylim(0, 1.15)
#     for bar, val in zip(b1, p_lp):
#         ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
#                 f"{val:.3f}", ha="center", fontsize=8)
#     for bar, val in zip(b2, p_tr):
#         ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
#                 f"{val:.3f}", ha="center", fontsize=8)
#     ax.set_title("Test d'adéquation : LP pure vs tronquée (vert = non rejetée à $p > 0{,}1$)")
#     ax.legend(loc="upper right", fontsize=9)
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_11_pvalues.pdf")
#     plt.close(fig)
#     print("[OK 11] pvalues")


# def graphe_12_sensibilite():
#     if not os.path.exists(LP_CSV_4JEUX):
#         print("[SKIP 12]")
#         return
#     df = pd.read_csv(LP_CSV_4JEUX)
#     agrege = df[df["population"] == "agrege"]
#     jeux_ordre = ["100m_300s", "100m_600s", "200m_1200s", "200m_1800s"]
#     labels_jeux = ["100m\n5 min", "100m\n10 min", "200m\n20 min", "200m\n30 min"]
#     idx = {j: i for i, j in enumerate(jeux_ordre)}
#     fig, ax = plt.subplots(figsize=(8, 5))
#     for var, marqueur, couleur, nom in [("sejour", "o", COUL_GARDE, "séjour"),
#                                           ("distance", "s", COUL_REF, "distance")]:
#         sub = agrege[agrege["variable"] == var]
#         xs = [idx[j] for j in jeux_ordre if j in set(sub["jeu"])]
#         ys = [float(sub[sub["jeu"] == j]["alpha"].iloc[0])
#               for j in jeux_ordre if j in set(sub["jeu"])]
#         es = [float(sub[sub["jeu"] == j]["sigma_alpha"].iloc[0])
#               for j in jeux_ordre if j in set(sub["jeu"])]
#         ax.errorbar(xs, ys, yerr=es, marker=marqueur, capsize=4,
#                     lw=1.5, color=couleur, label=nom)
#     ax.axhline(2, ls=":", color="gray", alpha=0.6)
#     ax.axhline(3, ls=":", color="gray", alpha=0.6)
#     ax.text(3.3, 2.05, "$\\alpha = 2$", fontsize=8, color="gray")
#     ax.text(3.3, 3.05, "$\\alpha = 3$", fontsize=8, color="gray")
#     ax.set_xticks(range(len(jeux_ordre))); ax.set_xticklabels(labels_jeux)
#     ax.set_xlabel("Jeu de seuils d'arrêt")
#     ax.set_ylabel("$\\alpha$ ajusté (agrégé)")
#     ax.set_title("Sensibilité de $\\alpha$ aux seuils d'arrêt "
#                  "(bande = $\\sigma_\\alpha$)")
#     ax.legend(loc="best", fontsize=9)
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_12_sensibilite.pdf")
#     plt.close(fig)
#     print("[OK 12] sensibilite")


# def graphe_13_ccdf_autres():
#     if not os.path.exists(LP_CSV_SEJOURS):
#         print("[SKIP 13]")
#         return
#     df_sej = pd.read_csv(LP_CSV_SEJOURS)
#     series = {
#         "distance": distances_entre_arrets(df_sej, meme_segment_seulement=True)["distance_m"].to_numpy(dtype=float),
#         "sejour_avec_censurees": df_sej["duree_s"].to_numpy(dtype=float),
#         "distance_a_cheval": distances_entre_arrets(df_sej, meme_segment_seulement=False)["distance_m"].to_numpy(dtype=float),
#     }
#     titres = {
#         "distance": "Distance (même segment)",
#         "sejour_avec_censurees": "Séjour (avec censurés)",
#         "distance_a_cheval": "Distance (à cheval)",
#     }
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
#     for ax, (var, vals) in zip(axes, series.items()):
#         vals = vals[np.isfinite(vals) & (vals > 0)]
#         p = _parametres_lp(var)
#         vals = vals[vals >= p["xmin"]]
#         x_emp, ccdf_emp = _ccdf_empirique(vals)
#         x_theo = np.logspace(np.log10(p["xmin"]), np.log10(x_emp.max()), 200)
#         ccdf_lp = _ccdf_lp(x_theo, p["alpha"], p["xmin"])
#         if np.isfinite(p["lambda_tronq"]) and p["lambda_tronq"] > 0:
#             ccdf_tr = _ccdf_tronquee(x_theo, p["alpha_tronq"], p["lambda_tronq"], p["xmin"])
#         else:
#             ccdf_tr = np.full_like(x_theo, np.nan)
#         ax.plot(x_emp, ccdf_emp, "o", ms=1.5, alpha=0.35, color=COUL_EXCLU)
#         ax.plot(x_theo, ccdf_lp, "-", lw=2, color=COUL_GARDE,
#                 label=f"LP $\\alpha={p['alpha']:.2f}$")
#         ax.plot(x_theo, ccdf_tr, "-", lw=2, color=COUL_REF,
#                 label=f"Tronq. $\\alpha={p['alpha_tronq']:.2f}$")
#         ax.axvline(p["xmin"], ls=":", color="k", alpha=0.5)
#         ax.set_xscale("log"); ax.set_yscale("log")
#         ax.set_title(titres[var], fontsize=10)
#         ax.set_xlabel("x")
#         ax.set_ylabel("$P(X \\geq x)$")
#         ax.legend(loc="lower left", fontsize=8)
#     fig.tight_layout()
#     sauvegarder(fig, f"{FIG_DIR}/graphe_13_ccdf_autres.pdf")
#     plt.close(fig)
#     print("[OK 13] ccdf_autres")


# # ── Main ────────────────────────────────────────────────────────────────────
# def main():
#     print("=" * 70)
#     print("Génération des graphes du mémoire")
#     print("=" * 70)

#     try:
#         df = charger_tous_segments()
#     except Exception as e:
#         print(f"[ERREUR] Impossible de charger resultats_H/*.csv : {e}")
#         df = None

#     # Validation fGn : quatre fichiers séparés
#     graphe_1a_validation_dfa()
#     graphe_1b_validation_rs_brut()
#     graphe_1c_validation_rs_corrige()
#     graphe_1d_validation_whittle()

#     if df is not None:
#         graphe_2_distribution(df)
#         graphe_3a_dfa_rs(df)
#         graphe_3b_dfa_wh(df)
#         graphe_3c_rs_wh(df)
#         graphe_7_longueurs(df)

#     graphe_4_user_ic()
#     graphe_5_heatmap()
#     graphe_6_tdrive()
#     graphe_8_deltas()
#     graphe_9_ccdf_sejour()
#     graphe_10_aicc()
#     graphe_11_pvalues()
#     graphe_12_sensibilite()
#     graphe_13_ccdf_autres()

#     print()
#     print("=" * 70)
#     print(f"Terminé. Fichiers dans {FIG_DIR}/")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# 06_graphes_1000.py  --  NE RIEN ECRASER

# Produit les 4 graphiques descriptifs des estimateurs de Hurst sur les
# segments >= 1000 points (issus de 05_segments_1000.py).

# Les fichiers sont ecrits dans figures/ avec des noms NEUFS (prefixe 14),
# qui ne rentrent en collision avec aucun fichier existant graphe_1..13.

# Sorties (PNG 150 dpi + PDF) :
#     figures/graphe_14a_boxplots_1000.{png,pdf}
#     figures/graphe_14b_hurst_vs_npoints_1000.{png,pdf}
#     figures/graphe_14c1_dfa_rs_1000.{png,pdf}
#     figures/graphe_14c2_dfa_wh_1000.{png,pdf}
#     figures/graphe_14c3_rs_wh_1000.{png,pdf}
#     figures/graphe_14d_medianes_user_ic_1000.{png,pdf}

# Usage :
#     python 06_graphes_1000.py
#     python 06_graphes_1000.py --segments autre/segments_1000.csv --sortie autre_dossier
# """

# import os
# import argparse
# from datetime import datetime

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# SEGMENTS_CSV = os.path.join("resultats_H_1000", "segments_1000.csv")
# SORTIE_DIR   = "figures"

# ESTIMATEURS = ["h_dfa", "h_rs", "h_whittle"]
# COULEURS = {"h_dfa": "#1f77b4", "h_rs": "#ff7f0e", "h_whittle": "#2ca02c"}
# LABELS   = {"h_dfa": "H_DFA", "h_rs": "H_RS", "h_whittle": "H_Whittle"}

# # --- Noms NEUFS (aucun conflit avec graphe_1..13 existants) --------------
# NOM_G1  = "graphe_14a_boxplots_1000"
# NOM_G2  = "graphe_14b_hurst_vs_npoints_1000"
# NOM_G3A = "graphe_14c1_dfa_rs_1000"
# NOM_G3B = "graphe_14c2_dfa_wh_1000"
# NOM_G3C = "graphe_14c3_rs_wh_1000"
# NOM_G4  = "graphe_14d_medianes_user_ic_1000"

# # Compat matplotlib : boxplot(vert=) déprécié depuis 3.9
# _MPL_VER = tuple(map(int, matplotlib.__version__.split(".")[:2]))

# def _boxplot(ax, data, horizontal=False, labels=None, **kw):
#     if _MPL_VER >= (3, 9):
#         return ax.boxplot(data,
#                           orientation=("horizontal" if horizontal else "vertical"),
#                           tick_labels=labels, **kw)
#     return ax.boxplot(data, vert=(not horizontal), labels=labels, **kw)


# # ----------------------------------------------------------------------
# def charger(chemin_seg):
#     df = pd.read_csv(chemin_seg)
#     for c in ESTIMATEURS + ["n_points", "divergence", "h_rs_corrige"]:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")
#     return df


# def _verifier_absence(dossier, nom):
#     """Refuse d'ecraser un fichier existant (securite)."""
#     for ext in (".png", ".pdf"):
#         p = os.path.join(dossier, nom + ext)
#         if os.path.exists(p):
#             raise SystemExit(
#                 f"[ABORT] Le fichier existe deja : {p}\n"
#                 f"        Ce script ne doit rien ecraser. "
#                 f"Renommez ou supprimez ce fichier avant de relancer.")


# def sauver(fig, dossier, nom):
#     os.makedirs(dossier, exist_ok=True)
#     _verifier_absence(dossier, nom)
#     p_png = os.path.join(dossier, nom + ".png")
#     p_pdf = os.path.join(dossier, nom + ".pdf")
#     fig.savefig(p_png, dpi=150, bbox_inches="tight")
#     fig.savefig(p_pdf, bbox_inches="tight")
#     plt.close(fig)
#     print(f"[OK] {p_png}  |  {p_pdf}")


# # ----------------------------------------------------------------------
# # Graphe 1 -> graphe_14a_boxplots_1000
# # ----------------------------------------------------------------------
# def graphe1_boxplots(df, dossier):
#     fig, ax = plt.subplots(figsize=(9, 6))
#     data = [df[c].dropna().values for c in ESTIMATEURS]

#     bp = _boxplot(ax, data, labels=[LABELS[c] for c in ESTIMATEURS],
#                   patch_artist=True, widths=0.55, showfliers=False)
#     for patch, c in zip(bp["boxes"], ESTIMATEURS):
#         patch.set_facecolor(COULEURS[c]); patch.set_alpha(0.40)
#     for m in bp["medians"]:
#         m.set_color("black"); m.set_linewidth(1.4)

#     rng = np.random.default_rng(42)
#     for i, (v, c) in enumerate(zip(data, ESTIMATEURS), 1):
#         if len(v):
#             jit = rng.uniform(-0.15, 0.15, size=len(v))
#             ax.scatter(np.full(len(v), i) + jit, v, s=4, alpha=0.15,
#                        color=COULEURS[c], edgecolors="none", rasterized=True)

#     ax.axhline(0.5, color="grey", ls="--", lw=1, label="H = 0.5")

#     ymin, ymax = ax.get_ylim()
#     ax.set_ylim(ymin, ymax + 0.15 * (ymax - ymin))
#     for i, v in enumerate(data, 1):
#         if len(v):
#             ax.text(i, ymax + 0.02 * (ymax - ymin),
#                     f"n={len(v)}\nméd={np.median(v):.3f}\nmoy={np.mean(v):.3f}",
#                     ha="center", va="bottom", fontsize=8)

#     ax.set_ylabel("Estimation de H")
#     ax.set_title(f"Distribution des trois estimateurs de Hurst\n"
#                  f"({len(df)} segments ≥ 1000 pts, {df['user_id'].nunique()} users)")
#     ax.legend(loc="lower right", fontsize=8)
#     ax.grid(True, alpha=0.3, axis="y")
#     sauver(fig, dossier, NOM_G1)


# # ----------------------------------------------------------------------
# # Graphe 2 -> graphe_14b_hurst_vs_npoints_1000
# # ----------------------------------------------------------------------
# def graphe2_hurst_vs_n(df, dossier):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
#     sub = df.dropna(subset=["n_points"])
#     ymin_global, ymax_global = 1e9, -1e9

#     for ax, c in zip(axes, ESTIMATEURS):
#         d = sub.dropna(subset=[c])
#         if not len(d):
#             ax.set_visible(False); continue

#         ax.scatter(d["n_points"], d[c], s=8, alpha=0.22,
#                    color=COULEURS[c], edgecolors="none", rasterized=True)

#         nmin = max(1000, float(d["n_points"].min()))
#         nmax = float(d["n_points"].max())
#         if nmax > nmin and len(d) > 20:
#             bins = np.logspace(np.log10(nmin), np.log10(nmax + 1), 12)
#             dd = d.assign(_b=pd.cut(d["n_points"], bins=bins))
#             med = dd.groupby("_b", observed=True)[c].agg(["median", "count"])
#             centres = np.array([iv.mid for iv in med.index])
#             ok = med["count"].values >= 5
#             ax.plot(centres[ok], med["median"].values[ok], "-o",
#                     color="black", lw=1.4, ms=4, label="médiane par tranche")

#         ax.set_xscale("log")
#         ax.set_xlabel("n_points (échelle log)")
#         ax.set_title(LABELS[c])
#         ax.axhline(0.5, color="grey", ls="--", lw=1)
#         ax.grid(True, alpha=0.3)
#         lo = float(np.nanmin(d[c].values)); hi = float(np.nanmax(d[c].values))
#         ymin_global = min(ymin_global, lo); ymax_global = max(ymax_global, hi)
#         if ax is axes[0]:
#             ax.set_ylabel("Estimation de H")

#     if ymin_global < ymax_global:
#         pad = 0.05 * (ymax_global - ymin_global)
#         axes[0].set_ylim(ymin_global - pad, ymax_global + pad)
#     axes[0].legend(loc="best", fontsize=8)

#     fig.suptitle("Effet de la longueur du segment sur l'estimation de H")
#     sauver(fig, dossier, NOM_G2)


# # ----------------------------------------------------------------------
# # Graphe 3 -> graphe_14c1 / 14c2 / 14c3
# # ----------------------------------------------------------------------
# def _graphe3_paire(df, a, b, nom, dossier):
#     d = df.dropna(subset=[a, b])
#     if not len(d):
#         print(f"[SKIP] {nom} : aucune donnée"); return

#     fig, ax = plt.subplots(figsize=(7, 6))
#     ax.scatter(d[a], d[b], s=8, alpha=0.22,
#                color="#222222", edgecolors="none", rasterized=True)

#     lo = float(min(d[a].min(), d[b].min()))
#     hi = float(max(d[a].max(), d[b].max()))
#     pad = 0.05 * (hi - lo); lo -= pad; hi += pad
#     ax.plot([lo, hi], [lo, hi], "--", color="red", lw=1, label="y = x")

#     k, c0 = np.polyfit(d[a], d[b], 1)
#     xs = np.linspace(lo, hi, 100)
#     ax.plot(xs, k * xs + c0, "-", color="blue", lw=1,
#             label=f"régression y = {k:.2f}x + {c0:.2f}")

#     r  = d[a].corr(d[b])
#     rs = d[a].corr(d[b], method="spearman")
#     ecart = (d[a] - d[b]).abs()

#     ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
#     ax.set_xlabel(LABELS[a]); ax.set_ylabel(LABELS[b])
#     ax.set_title(f"{LABELS[a]} vs {LABELS[b]}\n"
#                  f"Pearson r={r:.3f} | Spearman ρ={rs:.3f}")
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="upper left", fontsize=8)
#     ax.text(0.98, 0.02,
#             f"N={len(d)}\n|Δ| médiane={ecart.median():.3f}\n|Δ| moyen={ecart.mean():.3f}",
#             transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
#             bbox=dict(fc="white", ec="grey", alpha=0.85))

#     sauver(fig, dossier, nom)


# def graphe3_comparaisons(df, dossier):
#     _graphe3_paire(df, "h_dfa", "h_rs",      NOM_G3A, dossier)
#     _graphe3_paire(df, "h_dfa", "h_whittle", NOM_G3B, dossier)
#     _graphe3_paire(df, "h_rs",  "h_whittle", NOM_G3C, dossier)


# # ----------------------------------------------------------------------
# # Graphe 4 -> graphe_14d_medianes_user_ic_1000  (forest plot + IC 95 %)
# # ----------------------------------------------------------------------
# def _bootstrap_ci_median(v, n_boot=500, alpha=0.05, seed=42):
#     v = np.asarray(v); v = v[~np.isnan(v)]
#     if len(v) < 5:
#         return (np.nan, np.nan)
#     rng = np.random.default_rng(seed)
#     idx = rng.integers(0, len(v), size=(n_boot, len(v)))
#     meds = np.median(v[idx], axis=1)
#     return (float(np.percentile(meds, 100 * alpha / 2)),
#             float(np.percentile(meds, 100 * (1 - alpha / 2))))


# def graphe4_medianes_user_ic(df, dossier):
#     if "user_id" not in df.columns or df["user_id"].nunique() == 0:
#         print("[SKIP] graphe_14d : aucun user_id"); return

#     users = sorted(df["user_id"].dropna().unique())
#     n_users = len(users)
#     offsets = {"h_dfa": -0.22, "h_rs": 0.0, "h_whittle": +0.22}

#     stats = {}
#     for u in users:
#         sub = df[df["user_id"] == u]
#         for c in ESTIMATEURS:
#             v = sub[c].dropna().values
#             if len(v) == 0:
#                 stats[(u, c)] = (np.nan, np.nan, np.nan, 0); continue
#             med = float(np.median(v))
#             lo, hi = _bootstrap_ci_median(v)
#             stats[(u, c)] = (med, lo, hi, len(v))

#     fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * n_users + 2)))
#     y = np.arange(n_users)

#     for c in ESTIMATEURS:
#         for i, u in enumerate(users):
#             med, lo, hi, n = stats[(u, c)]
#             if np.isnan(med): continue
#             yy = i + offsets[c]
#             if not (np.isnan(lo) or np.isnan(hi)):
#                 ax.hlines(yy, lo, hi, color=COULEURS[c], lw=1.6, alpha=0.75)
#                 ax.plot([lo, hi], [yy, yy], "|", color=COULEURS[c], ms=6)
#             ax.plot(med, yy, "o", color=COULEURS[c], ms=6,
#                     label=LABELS[c] if i == 0 else "")

#     ax.axvline(0.5, color="grey", ls="--", lw=1, label="H = 0.5")
#     ax.set_yticks(y)
#     ax.set_yticklabels([f"{u}  (n={stats[(u, 'h_dfa')][3]})" for u in users],
#                        fontsize=8)
#     ax.invert_yaxis()
#     ax.set_xlabel("Médiane de H par utilisateur (barres = IC 95 % bootstrap)")
#     ax.set_title(f"Médianes et IC 95 % par utilisateur ({n_users} users, "
#                  f"segments ≥ 1000 pts)")
#     ax.legend(loc="lower right", fontsize=8)
#     ax.grid(True, alpha=0.3, axis="x")
#     sauver(fig, dossier, NOM_G4)


# # ----------------------------------------------------------------------
# def main():
#     p = argparse.ArgumentParser(description="4 graphes segments ≥ 1000 pts (sans écraser)")
#     p.add_argument("--segments", default=SEGMENTS_CSV)
#     p.add_argument("--sortie",   default=SORTIE_DIR)
#     args = p.parse_args()

#     if not os.path.exists(args.segments):
#         raise SystemExit(f"Fichier introuvable : {args.segments}")

#     df = charger(args.segments)
#     print(f"Chargé : {len(df)} segments, "
#           f"{df['user_id'].nunique()} utilisateurs\n")

#     graphe1_boxplots(df, args.sortie)
#     graphe2_hurst_vs_n(df, args.sortie)
#     graphe3_comparaisons(df, args.sortie)
#     graphe4_medianes_user_ic(df, args.sortie)
#     print("\nTerminé (aucun fichier existant écrasé).")


# if __name__ == "__main__":
#     debut = datetime.now()
#     main()
#     print(f"Duree : {(datetime.now()-debut).total_seconds():.1f} s")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Régénère UNIQUEMENT les 4 graphes de validation fGn (1a-1d)
avec la nouvelle table de validation (biais/RMSE, 200 réalisations).

Les données sont EMBARQUÉES dans ce script → aucun besoin préalable
du CSV resultats_H/validation/exactitude_resume.csv.
Le CSV est néanmoins ré-écrit au même format, pour cohérence avec
le pipeline existant (graphes 2-13 inchangés).

Sorties (par défaut) :
    memoire/figures/graphe_1a_validation_dfa.{pdf,png}
    memoire/figures/graphe_1b_validation_rs_brut.{pdf,png}
    memoire/figures/graphe_1c_validation_rs_corrige.{pdf,png}
    memoire/figures/graphe_1d_validation_whittle.{pdf,png}

Usage :
    python regenere_validation.py
    python regenere_validation.py --out figures
    python regenere_validation.py --csv ""        # ne pas ré-écrire le CSV
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════
# Données embarquées : {methode: {h0: [(n, biais, rmse), ...]}}
# Recopiées depuis la table de validation (fGn, 200 réal., graine 20261018)
# ═══════════════════════════════════════════════════════════════════════
_DATA = {
    "dfa": {
        0.1: [(300,+0.075,0.078),(500,+0.060,0.061),(1000,+0.047,0.048),(2000,+0.040,0.040),(5000,+0.031,0.032)],
        0.2: [(300,+0.058,0.066),(500,+0.048,0.053),(1000,+0.034,0.039),(2000,+0.029,0.033),(5000,+0.023,0.026)],
        0.3: [(300,+0.042,0.055),(500,+0.036,0.048),(1000,+0.027,0.035),(2000,+0.023,0.031),(5000,+0.019,0.025)],
        0.4: [(300,+0.033,0.056),(500,+0.026,0.048),(1000,+0.017,0.035),(2000,+0.016,0.029),(5000,+0.011,0.021)],
        0.5: [(300,+0.020,0.056),(500,+0.017,0.044),(1000,+0.012,0.037),(2000,+0.016,0.030),(5000,+0.012,0.024)],
        0.6: [(300,+0.020,0.060),(500,+0.013,0.050),(1000,+0.013,0.038),(2000,+0.012,0.028),(5000,+0.010,0.023)],
        0.7: [(300,+0.014,0.060),(500,+0.013,0.050),(1000,+0.002,0.038),(2000,+0.007,0.032),(5000,+0.007,0.025)],
        0.8: [(300,+0.007,0.065),(500,+0.006,0.057),(1000,+0.003,0.041),(2000,+0.006,0.034),(5000,+0.003,0.023)],
        0.9: [(300,+0.006,0.069),(500,+0.001,0.056),(1000,+0.004,0.046),(2000,+0.002,0.033),(5000,+0.004,0.028)],
    },
    "rs_brut": {
        0.1: [(300,+0.167,0.170),(500,+0.153,0.155),(1000,+0.143,0.144),(2000,+0.131,0.132),(5000,+0.119,0.120)],
        0.2: [(300,+0.143,0.150),(500,+0.136,0.139),(1000,+0.120,0.122),(2000,+0.109,0.111),(5000,+0.099,0.100)],
        0.3: [(300,+0.114,0.124),(500,+0.111,0.117),(1000,+0.099,0.103),(2000,+0.086,0.089),(5000,+0.078,0.080)],
        0.4: [(300,+0.086,0.100),(500,+0.084,0.097),(1000,+0.075,0.082),(2000,+0.066,0.072),(5000,+0.056,0.059)],
        0.5: [(300,+0.063,0.085),(500,+0.060,0.077),(1000,+0.049,0.059),(2000,+0.050,0.058),(5000,+0.042,0.049)],
        0.6: [(300,+0.040,0.073),(500,+0.036,0.066),(1000,+0.033,0.052),(2000,+0.028,0.041),(5000,+0.025,0.036)],
        0.7: [(300,+0.009,0.060),(500,+0.009,0.056),(1000,+0.001,0.042),(2000,+0.006,0.034),(5000,+0.005,0.028)],
        0.8: [(300,-0.031,0.071),(500,-0.027,0.068),(1000,-0.025,0.051),(2000,-0.023,0.044),(5000,-0.022,0.034)],
        0.9: [(300,-0.079,0.102),(500,-0.072,0.091),(1000,-0.062,0.074),(2000,-0.061,0.070),(5000,-0.052,0.060)],
    },
    "rs_corrige": {
        0.1: [(300,+0.096,0.101),(500,+0.088,0.092),(1000,+0.086,0.089),(2000,+0.081,0.082),(5000,+0.077,0.078)],
        0.2: [(300,+0.071,0.084),(500,+0.071,0.077),(1000,+0.063,0.067),(2000,+0.060,0.063),(5000,+0.056,0.058)],
        0.3: [(300,+0.043,0.064),(500,+0.046,0.060),(1000,+0.043,0.051),(2000,+0.037,0.043),(5000,+0.036,0.040)],
        0.4: [(300,+0.014,0.053),(500,+0.019,0.052),(1000,+0.018,0.040),(2000,+0.016,0.033),(5000,+0.013,0.024)],
        0.5: [(300,-0.008,0.058),(500,-0.005,0.049),(1000,-0.007,0.034),(2000,+0.001,0.028),(5000,+0.000,0.025)],
        0.6: [(300,-0.031,0.068),(500,-0.029,0.063),(1000,-0.023,0.046),(2000,-0.021,0.037),(5000,-0.017,0.031)],
        0.7: [(300,-0.062,0.086),(500,-0.056,0.079),(1000,-0.055,0.069),(2000,-0.043,0.055),(5000,-0.037,0.046)],
        0.8: [(300,-0.103,0.121),(500,-0.092,0.111),(1000,-0.081,0.093),(2000,-0.073,0.082),(5000,-0.064,0.069)],
        0.9: [(300,-0.151,0.164),(500,-0.137,0.148),(1000,-0.118,0.125),(2000,-0.111,0.116),(5000,-0.094,0.099)],
    },
    "whittle": {
        0.1: [(300,-0.045,0.081),(500,-0.051,0.073),(1000,-0.044,0.069),(2000,-0.036,0.054),(5000,-0.028,0.043)],
        0.2: [(300,-0.035,0.100),(500,-0.029,0.081),(1000,-0.025,0.065),(2000,-0.018,0.050),(5000,-0.009,0.036)],
        0.3: [(300,-0.029,0.096),(500,-0.011,0.081),(1000,-0.010,0.058),(2000,-0.004,0.046),(5000,-0.003,0.035)],
        0.4: [(300,-0.022,0.096),(500,-0.005,0.079),(1000,-0.010,0.060),(2000,-0.006,0.044),(5000,-0.007,0.035)],
        0.5: [(300,-0.013,0.094),(500,-0.008,0.070),(1000,-0.009,0.061),(2000,-0.001,0.048),(5000,+0.000,0.035)],
        0.6: [(300,+0.002,0.096),(500,-0.009,0.077),(1000,+0.007,0.061),(2000,-0.004,0.041),(5000,+0.004,0.036)],
        0.7: [(300,+0.002,0.084),(500,+0.002,0.074),(1000,-0.011,0.064),(2000,-0.001,0.045),(5000,-0.003,0.032)],
        0.8: [(300,-0.004,0.095),(500,+0.001,0.080),(1000,+0.003,0.058),(2000,+0.000,0.045),(5000,-0.001,0.032)],
        0.9: [(300,-0.002,0.089),(500,-0.010,0.072),(1000,+0.006,0.054),(2000,+0.002,0.043),(5000,+0.001,0.037)],
    },
}


def _df_validation():
    rows = []
    for methode, par_h0 in _DATA.items():
        for h0, cellules in par_h0.items():
            for n, biais, rmse in cellules:
                rows.append(dict(methode=methode, h0=h0, n=n,
                                 biais=biais, rmse=rmse))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Tracé — même logique que l'existant, mais ylim auto-ajusté
# ═══════════════════════════════════════════════════════════════════════
def _graphe_validation_methode(sub, titre, chemin_pdf, annoter_biais=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in sorted(sub["n"].unique()):
        s = sub[sub["n"] == n].sort_values("h0")
        ax.errorbar(s["h0"], s["h0"] + s["biais"], yerr=s["rmse"],
                    marker="o", capsize=3, label=f"N={n}")

    x = np.array([0.05, 0.95])
    ax.fill_between(x, x - 0.05, x + 0.05, alpha=0.15, color="green",
                    label="Biais acceptable $\\pm 0{,}05$")
    ax.plot(x, x, "k--", alpha=0.5, label="y = x")
    ax.axhline(0.5, color="C3", linestyle=":", lw=1)

    # ─── ylim calculé depuis les barres d'erreur (correction vs ancien code) ───
    lo = float((sub["h0"] + sub["biais"] - sub["rmse"]).min())
    hi = float((sub["h0"] + sub["biais"] + sub["rmse"]).max())
    marge = 0.05
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(min(lo, 0.05) - marge, max(hi, 1.0) + marge)

    ax.set_xlabel("$H_0$ (valeur vraie)")
    ax.set_ylabel("$\\hat H$ (estimation)")
    ax.set_title(titre)
    ax.legend(loc="upper left", fontsize=8)

    if annoter_biais:
        n_min = sub["n"].min()
        s09 = sub[(sub["n"] == n_min) & (sub["h0"] == sub["h0"].max())]
        if not s09.empty:
            b   = float(s09["biais"].iloc[0])
            h0  = float(s09["h0"].iloc[0])
            ax.annotate(f"biais = {b:+.3f}",
                        xy=(h0, h0 + b),
                        xytext=(0.7, 0.55), fontsize=9, color="C3",
                        arrowprops=dict(arrowstyle="->", color="C3"))

    fig.tight_layout()
    fig.savefig(chemin_pdf)
    fig.savefig(chemin_pdf.replace(".pdf", ".png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="memoire/figures",
                   help="dossier de sortie (defaut : memoire/figures)")
    p.add_argument("--csv", default="resultats_H/validation/exactitude_resume.csv",
                   help='CSV à ré-écrire (defaut : resultats_H/validation/exactitude_resume.csv ; '
                        'passer "" pour ne rien écrire)')
    args = p.parse_args()

    df = _df_validation()

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"[OK] CSV ré-écrit : {args.csv}  ({len(df)} lignes)")

    os.makedirs(args.out, exist_ok=True)

    specs = [
        ("dfa",        "Validation DFA sur fGn de $H_0$ connu",           "graphe_1a_validation_dfa",        False),
        ("rs_brut",    "Validation R/S brut sur fGn de $H_0$ connu",      "graphe_1b_validation_rs_brut",    False),
        ("rs_corrige", "Validation R/S corrigé sur fGn de $H_0$ connu",   "graphe_1c_validation_rs_corrige", True),
        ("whittle",    "Validation Whittle local sur fGn de $H_0$ connu", "graphe_1d_validation_whittle",    False),
    ]
    for methode, titre, nom, ann in specs:
        sub = df[df["methode"] == methode]
        if sub.empty:
            print(f"[SKIP {nom}] méthode {methode} absente"); continue
        chemin = os.path.join(args.out, nom + ".pdf")
        _graphe_validation_methode(sub, titre, chemin, annoter_biais=ann)
        print(f"[OK] {chemin}  +  {chemin.replace('.pdf','.png')}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()