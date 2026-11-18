#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualiser_lp.py

Trace les 5 figures du chapitre "lois de puissance" a partir des CSV existants,
sans relancer le pipeline.

  1. CCDF log-log de sejour avec fits LP pure et tronquee
  2. Poids AICc (barres empilees) sur les 4 variables
  3. Test d'adequation : p LP (Clauset) vs p tronquee, seuil 0.1
  4. Sensibilite alpha aux 4 jeux de seuils d'arret
  5. CCDF log-log des 3 autres variables (2x2 subplots)

Usage :
    python3.13 visualiser_lp.py                    # tous les graphes
    python3.13 visualiser_lp.py --graphe 1         # un seul
    python3.13 visualiser_lp.py --graphe 1 3       # plusieurs
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate

from lois_de_puissance.alternatives import _log_normalisation_puissance_tronquee
from pretraitement.arrets import distances_entre_arrets

CSV_AGREGE = "resultats_H/lois_de_puissance/lois_resume_rapide_avec_alt_B1000.csv"
CSV_SEJOURS = "resultats_H/lois_de_puissance/sejour_rapide_avec_alt_B1000.csv"
CSV_4JEUX = "resultats_H/lois_de_puissance/lois_resume.csv"
DOSSIER_IMAGES = "images"


# ----------------------------------------------------------------------
# Utilitaires numeriques
# ----------------------------------------------------------------------
def _ccdf_empirique(x):
    """CCDF empirique : P(X >= x_i) pour x tries croissants."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return x, 1.0 - np.arange(n) / n


def _ccdf_lp(x, alpha, xmin):
    """CCDF de la loi de puissance pure : (x/xmin)^(1-alpha), x >= xmin."""
    return (x / xmin) ** (1.0 - alpha)


def _ccdf_tronquee(x, alpha, lam, xmin):
    """CCDF de la tronquee : 1 - F(x), calculee par integration de xmin a x.
    L'integrale de xmin a x est sur un intervalle fini, donc numeriquement stable,
    contrairement a l'integrale de x a l'infini qui echoue pour alpha > 2 et lambda petit."""
    log_Z = _log_normalisation_puissance_tronquee(alpha, lam, xmin)
    if not np.isfinite(log_Z):
        return np.full_like(x, np.nan)
    Z = np.exp(log_Z)
    out = np.empty(len(x))
    for i, xi in enumerate(x):
        if xi < xmin:
            out[i] = np.nan
            continue
        val, _ = integrate.quad(lambda t: t ** (-alpha) * np.exp(-lam * t),
                                xmin, xi, limit=100)
        out[i] = 1.0 - val / Z
    return out


def _filtre_sejour(df):
    """Durees de sejour non censurees (interne uniquement), comme le pipeline LP."""
    d = df.copy()
    d["censuree"] = d["censuree"].astype(str).str.lower()
    mask = (d["type"] == "interne") & (d["censuree"] == "false")
    return d.loc[mask, "duree_s"].to_numpy(dtype=float)


def _filtre_sejour_tous(df):
    """Toutes les durees (censures + entre_segments), pour sejour_avec_censurees."""
    return df["duree_s"].to_numpy(dtype=float)


def _parametres_lp(variable, csv=CSV_AGREGE):
    """Extrait xmin, alpha, alpha_tronq, lambda_tronq pour la ligne agregee."""
    df = pd.read_csv(csv)
    ligne = df[(df["population"] == "agrege") & (df["variable"] == variable)].iloc[0]
    params = json.loads(ligne["parametres_alternatives"]) if ligne["parametres_alternatives"] else {}
    tronq = params.get("puissance_tronquee", {})
    return dict(
        xmin=float(ligne["xmin"]),
        alpha=float(ligne["alpha"]),
        alpha_tronq=float(tronq["alpha"]) if "alpha" in tronq else np.nan,
        lambda_tronq=float(tronq["lambda"]) if "lambda" in tronq else np.nan,
        n_queue=int(ligne["n_queue"]),
        p_bootstrap=float(ligne["p_bootstrap"]),
        p_tronq=float(ligne["p_adequation_tronquee"]),
    )


# ----------------------------------------------------------------------
# Graphe 1 : CCDF de sejour avec fits LP et tronquee
# ----------------------------------------------------------------------
def graphe_1(fichier=os.path.join(DOSSIER_IMAGES, "lp_ccdf_sejour.png")):
    durees = _filtre_sejour(pd.read_csv(CSV_SEJOURS))
    p = _parametres_lp("sejour")
    x_emp, ccdf_emp = _ccdf_empirique(durees)

    x_theo = np.logspace(np.log10(p["xmin"]), np.log10(x_emp.max()), 300)
    ccdf_lp = _ccdf_lp(x_theo, p["alpha"], p["xmin"])
    ccdf_tr = _ccdf_tronquee(x_theo, p["alpha_tronq"], p["lambda_tronq"], p["xmin"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(x_emp, ccdf_emp, "o", ms=2, alpha=0.4, color="gray", label=f"Observé (n={len(durees)})")
    ax.plot(x_theo, ccdf_lp, "-", lw=2, color="C0",
            label=f"LP pure ($\\alpha={p['alpha']:.2f}$)")
    ax.plot(x_theo, ccdf_tr, "-", lw=2, color="C3",
            label=f"Tronquée ($\\alpha={p['alpha_tronq']:.2f}$, $\\lambda={p['lambda_tronq']:.1e}$)")
    ax.axvline(p["xmin"], ls=":", color="k", alpha=0.5)
    ax.text(p["xmin"], ax.get_ylim()[1] * 0.7, " $x_{\\min}$", fontsize=9)
    echelle_coupure = 1.0 / p["lambda_tronq"] if p["lambda_tronq"] > 0 else np.inf
    ax.axvline(echelle_coupure, ls="--", color="C3", alpha=0.5)
    ax.text(echelle_coupure, ax.get_ylim()[1] * 0.3,
            f" $1/\\lambda\\approx{echelle_coupure/3600:.1f}$ h", fontsize=9, color="C3")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Durée de séjour (s)")
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title("CCDF des durées de séjour (agrégé, x ≥ xmin)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)
    print(f"  ecrit : {fichier}")


# ----------------------------------------------------------------------
# Graphe 2 : poids AICc empiles
# ----------------------------------------------------------------------
def graphe_2(fichier=os.path.join(DOSSIER_IMAGES, "lp_aicc.png")):
    df = pd.read_csv(CSV_AGREGE)
    agrege = df[df["population"] == "agrege"].copy()
    ordre_var = ["sejour", "distance", "sejour_avec_censurees", "distance_a_cheval"]
    labels = ["sejour", "distance\n(meme seg.)", "sejour\n(censures)", "distance\n(a cheval)"]
    agrege = agrege.set_index("variable").loc[ordre_var]
    modeles = ["loi_de_puissance", "exponentielle", "lognormale", "weibull", "puissance_tronquee"]
    noms = ["LP", "Exp", "Log-norm", "Weibull", "Tronquee"]
    couleurs = ["C0", "C1", "C2", "C4", "C3"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bas = np.zeros(len(agrege))
    for m, nom, c in zip(modeles, noms, couleurs):
        vals = agrege[f"poids_aicc_{m}"].to_numpy(dtype=float)
        ax.bar(labels, vals, bottom=bas, label=nom, color=c, edgecolor="white", lw=0.5)
        bas += vals
    ax.set_ylabel("Poids AICc")
    ax.set_ylim(0, 1.02)
    ax.set_title("Poids AICc des 5 modèles, par variable (agrégé)")
    ax.legend(loc="upper right", ncol=5, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)
    print(f"  ecrit : {fichier}")


# ----------------------------------------------------------------------
# Graphe 3 : p-valeurs comparees LP vs tronquee
# ----------------------------------------------------------------------
def graphe_3(fichier=os.path.join(DOSSIER_IMAGES, "lp_pvalues.png")):
    df = pd.read_csv(CSV_AGREGE)
    agrege = df[df["population"] == "agrege"].copy()
    ordre_var = ["sejour", "distance", "sejour_avec_censurees", "distance_a_cheval"]
    labels = ["sejour", "distance\n(meme seg.)", "sejour\n(censures)", "distance\n(a cheval)"]
    agrege = agrege.set_index("variable").loc[ordre_var]

    x = np.arange(len(agrege))
    largeur = 0.38
    p_lp = agrege["p_bootstrap"].to_numpy(dtype=float)
    p_tr = agrege["p_adequation_tronquee"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - largeur / 2, p_lp, largeur, label="LP pure (Clauset)",
                color=["C2" if p > 0.1 else "C3" for p in p_lp])
    b2 = ax.bar(x + largeur / 2, p_tr, largeur, label="Tronquée (bootstrap)",
                color=["C2" if p > 0.1 else "C3" for p in p_tr], alpha=0.7)
    ax.axhline(0.1, ls="--", color="k", label="seuil 0.1")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("p-valeur")
    ax.set_ylim(0, 1.15)

    # annotation des valeurs
    for bar, val in zip(b1, p_lp):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.3f}", ha="center", fontsize=8)
    for bar, val in zip(b2, p_tr):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.3f}", ha="center", fontsize=8)

    ax.set_title("Test d'adéquation : LP pure vs tronquée (vert = non rejetée à p > 0.1)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)
    print(f"  ecrit : {fichier}")


# ----------------------------------------------------------------------
# Graphe 4 : alpha vs seuils d'arret
# ----------------------------------------------------------------------
def graphe_4(fichier=os.path.join(DOSSIER_IMAGES, "lp_alpha_seuils.png")):
    df = pd.read_csv(CSV_4JEUX)
    agrege = df[df["population"] == "agrege"]
    jeux_ordre = ["100m_300s", "100m_600s", "200m_1200s", "200m_1800s"]
    labels_jeux = ["100m\n5 min", "100m\n10 min", "200m\n20 min", "200m\n30 min"]
    idx = {j: i for i, j in enumerate(jeux_ordre)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for var, marqueur, couleur, nom in [("sejour", "o", "C0", "sejour"),
                                          ("distance", "s", "C3", "distance")]:
        sub = agrege[agrege["variable"] == var]
        xs = [idx[j] for j in jeux_ordre if j in set(sub["jeu"])]
        ys = [float(sub[sub["jeu"] == j]["alpha"].iloc[0]) for j in jeux_ordre if j in set(sub["jeu"])]
        es = [float(sub[sub["jeu"] == j]["sigma_alpha"].iloc[0]) for j in jeux_ordre if j in set(sub["jeu"])]
        ax.errorbar(xs, ys, yerr=es, marker=marqueur, capsize=4, lw=1.5, color=couleur, label=nom)

    ax.axhline(2, ls=":", color="gray", alpha=0.6)
    ax.axhline(3, ls=":", color="gray", alpha=0.6)
    ax.text(3.3, 2.05, "alpha = 2", fontsize=8, color="gray")
    ax.text(3.3, 3.05, "alpha = 3", fontsize=8, color="gray")
    ax.set_xticks(range(len(jeux_ordre))); ax.set_xticklabels(labels_jeux)
    ax.set_xlabel("Jeu de seuils d'arrêt")
    ax.set_ylabel(r"$\alpha$ ajusté (agrégé)")
    ax.set_title("Sensibilité de $\\alpha$ aux seuils d'arrêt (bande = $\\sigma_\\alpha$)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)
    print(f"  ecrit : {fichier}")


# ----------------------------------------------------------------------
# Graphe 5 : CCDF des 3 autres variables (2x2 subplots)
# ----------------------------------------------------------------------
def graphe_5(fichier=os.path.join(DOSSIER_IMAGES, "lp_ccdf_autres.png")):
    df_sej = pd.read_csv(CSV_SEJOURS)
    series = {
        "distance": distances_entre_arrets(df_sej, meme_segment_seulement=True)["distance_m"].to_numpy(dtype=float),
        "sejour_avec_censurees": _filtre_sejour_tous(df_sej),
        "distance_a_cheval": distances_entre_arrets(df_sej, meme_segment_seulement=False)["distance_m"].to_numpy(dtype=float),
    }
    titres = {
        "distance": "Distance (même segment)",
        "sejour_avec_censurees": "Séjour (avec censurés)",
        "distance_a_cheval": "Distance (à cheval)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (var, vals) in zip(axes, series.items()):
        vals = vals[np.isfinite(vals) & (vals > 0)]
        p = _parametres_lp(var)
        x_emp, ccdf_emp = _ccdf_empirique(vals)

        x_theo = np.logspace(np.log10(p["xmin"]), np.log10(x_emp.max()), 200)
        ccdf_lp = _ccdf_lp(x_theo, p["alpha"], p["xmin"])
        if np.isfinite(p["lambda_tronq"]) and p["lambda_tronq"] > 0:
            ccdf_tr = _ccdf_tronquee(x_theo, p["alpha_tronq"], p["lambda_tronq"], p["xmin"])
        else:
            ccdf_tr = np.full_like(x_theo, np.nan)

        ax.plot(x_emp, ccdf_emp, "o", ms=1.5, alpha=0.35, color="gray")
        ax.plot(x_theo, ccdf_lp, "-", lw=2, color="C0", label=f"LP $\\alpha={p['alpha']:.2f}$")
        ax.plot(x_theo, ccdf_tr, "-", lw=2, color="C3", label=f"Tronq. $\\alpha={p['alpha_tronq']:.2f}$")
        ax.axvline(p["xmin"], ls=":", color="k", alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(titres[var], fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("P(X ≥ x)")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)
    print(f"  ecrit : {fichier}")


# ----------------------------------------------------------------------
# Point d'entree
# ----------------------------------------------------------------------
GRAPHE_FONCTIONS = {1: graphe_1, 2: graphe_2, 3: graphe_3, 4: graphe_4, 5: graphe_5}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--graphe", nargs="+", default=["all"],
                    help="numeros de graphes a tracer (1-5), ou 'all'")
    args = ap.parse_args()

    os.makedirs(DOSSIER_IMAGES, exist_ok=True)

    numeros = list(range(1, 6)) if "all" in args.graphe else [int(x) for x in args.graphe]
    for n in numeros:
        if n not in GRAPHE_FONCTIONS:
            print(f"[ERREUR] graphe {n} inconnu (1-5)")
            continue
        print(f"Graphe {n}...")
        try:
            GRAPHE_FONCTIONS[n]()
        except Exception as e:
            print(f"  [ERREUR] {e}")


if __name__ == "__main__":
    raise SystemExit(main())