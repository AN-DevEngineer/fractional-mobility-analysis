#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiences/tests_berkes_qu.py

Deux tests aux hypotheses nulles OPPOSEES, pour distinguer persistance apparente et memoire longue :
  - Berkes, Horvath, Kokoszka & Shao (2006) : H0 = memoire courte + rupture(s) de moyenne ; rejet = memoire longue.
  - Qu (2011) : H0 = vraie memoire longue stationnaire ; rejet = memoire longue spurieuse (regimes, tendance).

Deux modes :
  --simulation   taux de rejet (taille et puissance) sur series de nature CONNUE. A faire AVANT d'interpreter le reel :
                 un test dont on ne connait pas la taille et la puissance a n ~ 500-2000 ne prouve rien.
  --reel         applique les deux tests a la serie d_i de chaque segment de sortie/ ; ecrit un CSV par segment
                 et un resume (part de rejets, lecture croisee, par strate de longueur).

Exemples :
  python experiences/tests_berkes_qu.py --simpython experiences/tests_berkes_qu.py --reel --dossier sortie --sortie resultats_H/tests_berkes_quulation --n-rep 300 --sortie resultats_H/tests_berkes_qu
  python experiences/tests_berkes_qu.py --reel --dossier sortie --sortie resultats_H/tests_berkes_qu
"""

import argparse
import csv
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import GRAINE
from simulation import bruit_gaussien_fractionnaire, ar1, changements_de_regime, on_off
from validation.berkes import test_berkes
from validation.qu import test_qu, decision_croisee


# ------------------------------------------------------------------ modeles de serie (nature connue)
def _avec_rupture(base, n, rng, delta=1.0, nb=1):
    """Ajoute `nb` sauts de moyenne d'amplitude delta (en ecart-type du bruit) a des instants tires uniformement
    dans [0.2 n, 0.8 n], sans que deux ruptures soient a moins de 0.1 n l'une de l'autre."""
    x = base.copy()
    instants = []
    while len(instants) < nb:
        k = int(rng.integers(int(0.2 * n), int(0.8 * n)))
        if all(abs(k - j) > 0.1 * n for j in instants):
            instants.append(k)
    for k in instants:
        x[k:] += delta * rng.choice([-1.0, 1.0])
    return x


def catalogue():
    """
    nom -> (f(n, rng), nature) ; nature = verite pour chaque test :
      berkes_h0 : H0 de Berkes vraie (memoire courte, 0 ou quelques ruptures) -> le rejet est une ERREUR de type I
      qu_h0     : H0 de Qu vraie (memoire longue stationnaire)                -> le rejet est une ERREUR de type I
    """
    def fgn_ar(h):
        return lambda n, rng: bruit_gaussien_fractionnaire(n, h, rng) + 0.7 * ar1(n, 0.5, rng)

    def tendance(n, rng):
        t = np.arange(n) / n
        return 3.0 * np.sin(4.0 * np.pi * t) + rng.standard_normal(n)

    return {
        # --- memoire courte (H0 de Berkes vraie ; la persistance est apparente)
        "iid":                        (lambda n, rng: rng.standard_normal(n), "berkes_h0"),
        "ar1_0.5":                    (lambda n, rng: ar1(n, 0.5, rng), "berkes_h0"),
        "ar1_0.8":                    (lambda n, rng: ar1(n, 0.8, rng), "berkes_h0"),
        "iid + 1 rupture":            (lambda n, rng: _avec_rupture(rng.standard_normal(n), n, rng, 1.0, 1), "berkes_h0"),
        "ar1_0.5 + 1 rupture":        (lambda n, rng: _avec_rupture(ar1(n, 0.5, rng), n, rng, 1.0, 1), "berkes_h0"),
        "ar1_0.5 + 2 ruptures":       (lambda n, rng: _avec_rupture(ar1(n, 0.5, rng), n, rng, 1.0, 2), "berkes_h0"),
        "ar1_0.5 + 3 ruptures":       (lambda n, rng: _avec_rupture(ar1(n, 0.5, rng), n, rng, 1.0, 3), "berkes_h0"),
        "regimes (3 niveaux, ~60 pas)": (lambda n, rng: changements_de_regime(n, rng), "berkes_h0"),
        "tendance lisse + bruit":     (tendance, "berkes_h0"),
        # --- memoire longue stationnaire (H0 de Qu vraie)
        "fGn H=0.6 (d=0.1)":          (lambda n, rng: bruit_gaussien_fractionnaire(n, 0.6, rng), "qu_h0"),
        "fGn H=0.7 (d=0.2)":          (lambda n, rng: bruit_gaussien_fractionnaire(n, 0.7, rng), "qu_h0"),
        "fGn H=0.8 (d=0.3)":          (lambda n, rng: bruit_gaussien_fractionnaire(n, 0.8, rng), "qu_h0"),
        "fGn H=0.7 + AR(1) 0.5":      (fgn_ar(0.7), "qu_h0"),
        # --- memoire longue + rupture (alternative de Qu, aussi hors H0 de Berkes)
        "fGn H=0.7 + 1 rupture":      (lambda n, rng: _avec_rupture(bruit_gaussien_fractionnaire(n, 0.7, rng), n, rng, 1.0, 1), "mixte"),
        # --- ON/OFF a durees de Pareto (H theorique 0.75 pour alpha = 1.5)
        "ON/OFF Pareto alpha=1.5":    (lambda n, rng: on_off(n, 1.5, 1.5, rng), "qu_h0"),
    }


COLONNES_TESTS = [
    ("berkes u<=1", dict(u_max=1)), ("berkes u<=2", dict(u_max=2)), ("berkes u<=3", dict(u_max=3)),
    ("berkes u<=1 q=cube", dict(u_max=1, regle_q="cube")), ("berkes u<=1 q=log15", dict(u_max=1, regle_q="log15")),
]


def appliquer_les_tests(x, alpha=0.05):
    """Un dict de verdicts (True = rejet de H0 ; None = non concluant) pour une serie."""
    v = {nom: test_berkes(x, alpha=alpha, **kw)["rejette"] for nom, kw in COLONNES_TESTS}
    v["qu eps=0.02 +prewhit"] = test_qu(x, eps=0.02, alpha=alpha)["rejette"]
    v["qu eps=0.05 +prewhit"] = test_qu(x, eps=0.05, alpha=alpha)["rejette"]
    v["qu eps=0.02 brut"] = test_qu(x, eps=0.02, alpha=alpha, avec_preblanchiment=False)["rejette"]
    return v


# ------------------------------------------------------------------ mode simulation
def mode_simulation(args):
    os.makedirs(os.path.dirname(args.sortie) or ".", exist_ok=True)
    cat = catalogue()
    lignes = []
    t0 = time.time()
    for n in args.n:
        for nom, (f, nature) in cat.items():
            compte, valides = {}, {}
            for r in range(args.n_rep):
                rng = np.random.default_rng(np.random.SeedSequence([GRAINE, zlib.crc32(nom.encode()), n, r]))
                try:
                    verdicts = appliquer_les_tests(f(n, rng), args.alpha)
                except Exception as e:                       # une serie degeneree ne doit pas tuer l'experience
                    print(f"  [AVERTISSEMENT] {nom} n={n} rep={r} : {type(e).__name__}: {e}")
                    continue
                for k, v in verdicts.items():
                    if v is None:
                        continue
                    compte[k] = compte.get(k, 0) + int(v)
                    valides[k] = valides.get(k, 0) + 1
            for k in compte:
                lignes.append(dict(n=n, modele=nom, nature=nature, test=k,
                                   taux_rejet=compte[k] / valides[k], n_valides=valides[k]))
            print(f"[{time.time() - t0:6.0f}s] n={n:5d}  {nom:30s} " +
                  "  ".join(f"{k.split()[0][:2]}{k.split()[1][:6]}={compte[k] / valides[k]:.2f}" for k in list(compte)[:3] +
                            ["qu eps=0.02 +prewhit"] if "qu eps=0.02 +prewhit" in compte), flush=True)
    with open(args.sortie + "_simulation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "modele", "nature", "test", "taux_rejet", "n_valides"])
        w.writeheader()
        w.writerows(lignes)
    print("ecrit :", args.sortie + "_simulation.csv")


# ------------------------------------------------------------------ mode reel
def mode_reel(args):
    if args.npz:
        from outils.exporter_pour_analyse import lire_series_npz
        segments = lire_series_npz(args.npz)
    else:
        from outils.chargement_segments import tous_les_segments
        segments = tous_les_segments(args.dossier, args.max_utilisateurs)
    os.makedirs(os.path.dirname(args.sortie) or ".", exist_ok=True)
    chemin = args.sortie + "_reel.csv"
    champs = ["utilisateur", "segment_id", "n", "berkes_m_u1", "berkes_seuil_u1", "berkes_rejette_u1",
              "berkes_rejette_u2", "berkes_rejette_u3", "berkes_n_ruptures_u3", "qu_w_02", "qu_rejette_02",
              "qu_rejette_05", "qu_d", "qu_m", "qu_prewhit_p", "qu_prewhit_q", "decision_u1", "decision_u3"]
    n_ok = 0
    t0 = time.time()
    with open(chemin, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=champs)
        w.writeheader()
        for seg in segments:
            x = seg["d"]
            b1 = test_berkes(x, u_max=1, alpha=args.alpha)
            b2 = test_berkes(x, u_max=2, alpha=args.alpha)
            b3 = test_berkes(x, u_max=3, alpha=args.alpha)
            q02 = test_qu(x, eps=0.02, alpha=args.alpha)
            q05 = test_qu(x, eps=0.05, alpha=args.alpha)
            mod = q02["modele_prewhitening"]
            w.writerow(dict(
                utilisateur=seg["utilisateur"], segment_id=seg["segment_id"], n=len(x),
                berkes_m_u1=b1["m_stat"], berkes_seuil_u1=b1["seuil"], berkes_rejette_u1=b1["rejette"],
                berkes_rejette_u2=b2["rejette"], berkes_rejette_u3=b3["rejette"],
                berkes_n_ruptures_u3=b3["n_ruptures_estimees"],
                qu_w_02=q02["w"], qu_rejette_02=q02["rejette"], qu_rejette_05=q05["rejette"],
                qu_d=q02["d"], qu_m=q02["m"], qu_prewhit_p=None if mod is None else mod[0],
                qu_prewhit_q=None if mod is None else mod[1],
                decision_u1=decision_croisee(b1["rejette"], q02["rejette"]),
                decision_u3=decision_croisee(b3["rejette"], q02["rejette"])))
            n_ok += 1
            if n_ok % 200 == 0:
                print(f"[{time.time() - t0:6.0f}s] {n_ok} segments traites", flush=True)
    print(f"ecrit : {chemin} ({n_ok} segments)")
    resumer_reel(chemin)


def resumer_reel(chemin):
    """Imprime les parts de rejet et la lecture croisee, globalement et par strate de longueur."""
    import pandas as pd
    df = pd.read_csv(chemin)
    df["strate"] = pd.cut(df["n"], [0, 500, 1000, 10 ** 9], right=False, labels=["[300,500)", "[500,1000)", ">=1000"])
    print("\n=== Part de rejets (hors non concluants) ===")
    for col in ["berkes_rejette_u1", "berkes_rejette_u2", "berkes_rejette_u3", "qu_rejette_02", "qu_rejette_05"]:
        s = df[col].dropna().astype(bool)
        print(f"{col:22s} {s.mean():6.1%}  (n={len(s)})")
    print("\n=== Lecture croisee (Berkes u<=3 x Qu eps=0.02) ===")
    print(df["decision_u3"].value_counts(normalize=True).round(3).to_string())
    print("\n=== Par strate de longueur (part de rejets) ===")
    print(df.groupby("strate", observed=True)[["berkes_rejette_u3", "qu_rejette_02"]].mean().round(3).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--simulation", action="store_true")
    g.add_argument("--reel", action="store_true")
    g.add_argument("--resumer", metavar="CSV", help="resume d'un CSV *_reel.csv deja calcule")
    p.add_argument("--n", type=int, nargs="+", default=[500, 1000, 2000])
    p.add_argument("--n-rep", type=int, default=300)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dossier", default="sortie")
    p.add_argument("--npz", default=None, help="dossier export/ de outils/exporter_pour_analyse.py (remplace --dossier)")
    p.add_argument("--max-utilisateurs", type=int, default=None)
    p.add_argument("--sortie", default="resultats_H/tests_berkes_qu")
    a = p.parse_args()
    if a.simulation:
        mode_simulation(a)
    elif a.reel:
        mode_reel(a)
    else:
        resumer_reel(a.resumer)


if __name__ == "__main__":
    main()
