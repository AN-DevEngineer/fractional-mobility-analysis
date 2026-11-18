#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiences/mecanisme_on_off.py        (hypothese H2)

Hypothese : la memoire longue apparente de d_i vient de l'ALTERNANCE mouvement / arret dont les
durees ont une queue en loi de puissance. Pour un exposant de queue alpha dans ]1, 2[, un processus
ON/OFF a durees de Pareto a H = (3 - alpha) / 2 (Taqqu, Willinger & Sherman, 1997 -- formule a
confronter a la source avant publication).

PARTIE 1 (ce script, sans donnees) : verifier que NOS estimateurs retrouvent cette valeur theorique
sur des series ON/OFF simulees. Si DFA ou Whittle ne la retrouvent pas, on ne pourra pas utiliser
cette relation sur les donnees reelles : mieux vaut le savoir avant.

PARTIE 2 (donnees reelles, option --reel) : pour chaque utilisateur, on part de l'exposant alpha des temps
de sejour mesure par experiences/lois_de_puissance.py (methode de Clauset) et on le confronte a H.
Trois precautions, dans cet ordre :

  a. DOMAINE DE VALIDITE. Notre alpha est l'exposant de la DENSITE ; la formule H = (3 - a) / 2 porte
     sur l'exposant de SURVIE a = alpha - 1 et n'est valable que pour 1 < a < 2, soit 2 < alpha < 3.
     Les exposants de densite publies pour les temps de sejour sont plutot de 1.2 a 2.0 : beaucoup
     d'utilisateurs peuvent etre HORS domaine. Le script commence par compter combien sont dans le
     domaine ; s'ils sont hors domaine, la prediction ne s'applique pas et on le rapporte tel quel.
  b. LOI DE PUISSANCE PLAUSIBLE. On ne garde que les utilisateurs dont le verdict de Clauset est
     "plausible" : sinon l'exposant n'a pas de sens.
  c. COMPARAISON A LA MEME LONGUEUR. DFA sous-estime H sur des series courtes (voir la partie 1 a
     N = 500). On compare donc H observe NON SEULEMENT a la valeur theorique mais aussi a H SIMULE
     sur un ON/OFF de meme exposant et de meme longueur que les segments de l'utilisateur.

Hypotheses simplificatrices du modele simule (a rappeler dans le memoire) : seuls les arrets (etat OFF)
sont mesures ; les durees de deplacement (etat ON) sont supposees a queue plus legere (exposant de
survie 3) ; les deux etats partagent la meme duree minimale ; on simule l'indicateur 0/1
mouvement/arret, pas les distances d_i elles-memes. Un desaccord est un resultat a rapporter, pas un echec.

Usage :  python experiences/mecanisme_on_off.py --n-rep 60            # partie 1 (sans donnees)
         python experiences/mecanisme_on_off.py --reel                 # partie 2 (apres lois_de_puissance.py et 03_estimation_H.py)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from hurst.dfa import dfa_h
from hurst.rs import rs_complet
from hurst.whittle import whittle_local
from simulation.on_off_pareto import h_theorique_on_off, on_off


def verifier_theorie(alphas=(1.2, 1.4, 1.6, 1.8), n=3000, nb_rep=60, rng=None):
    """H estime sur des ON/OFF simules, compare a (3 - alpha)/2. Une ligne par alpha."""
    rng = rng or np.random.default_rng(config.GRAINE)
    lignes = []
    for alpha in alphas:
        h = {"dfa": [], "rs_corrige": [], "whittle": []}
        for _ in range(nb_rep):
            x = on_off(n, alpha, alpha, rng)
            if x.std() == 0:
                continue
            h["dfa"].append(dfa_h(x))
            h["rs_corrige"].append(rs_complet(x).h_corrige)
            h["whittle"].append(whittle_local(x).h)
        theorie = h_theorique_on_off(alpha, alpha)
        ligne = dict(alpha=alpha, n=n, nb_rep=len(h["dfa"]), h_theorique=theorie)
        for m, v in h.items():
            v = np.asarray(v, float)
            ligne[f"{m}_moyen"] = float(np.nanmean(v))
            ligne[f"{m}_ecart_type"] = float(np.nanstd(v, ddof=1))
            ligne[f"{m}_biais"] = float(np.nanmean(v) - theorie)
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def partie_reelle(chemin_lois=os.path.join("resultats_H", "lois_de_puissance", "lois_resume.csv"),
                  dossier_resultats_h="resultats_H", jeu=None, nb_rep=30, rng=None):
    """
    Compare H observe a H predit par le modele ON/OFF, utilisateur par utilisateur.

    Entree :
        chemin_lois          : lois_resume.csv produit par experiences/lois_de_puissance.py
        dossier_resultats_h  : dossier des <user>_H.csv produits par 03_estimation_H.py
        jeu                  : jeu de seuils d'arret a utiliser (defaut : celui de config.py)
        nb_rep               : repetitions de la simulation calibree a la meme longueur

    Sortie : DataFrame, une ligne par utilisateur retenu (vide s'il n'y en a aucun) :
        alpha, alpha_survie, xmin_s   exposant de la densite, de la survie, seuil de la queue (s)
        h_theorique                   (3 - alpha_survie) / 2
        h_simule_meme_N               H (DFA) moyen sur des ON/OFF simules de meme exposant et longueur
        h_observe                     mediane de h_dfa sur les segments de l'utilisateur
        ecart_obs_theorie, ecart_obs_simule
    Ecrit aussi un compte rendu a l'ecran : combien d'utilisateurs sont dans le domaine de validite.
    """
    jeu = jeu or f"{config.ARRET_RAYON_M:g}m_{config.ARRET_DUREE_MIN_S:g}s"
    if not os.path.exists(chemin_lois):
        print(f"[ERREUR] {chemin_lois} introuvable : lancez d'abord experiences/lois_de_puissance.py")
        return pd.DataFrame()
    lois = pd.read_csv(chemin_lois, dtype={"population": str})
    sejour = lois[(lois["jeu"] == jeu) & (lois["variable"] == "sejour")
                  & (lois["population"] != "agrege") & lois["alpha"].notna()]
    print(f"\nPARTIE 2 -- jeu de seuils {jeu}")
    if sejour.empty:
        print("Aucun utilisateur n'a assez de sejours pour un ajustement.")
        return pd.DataFrame()

    dans_domaine = sejour[sejour["dans_domaine_on_off"].astype(bool)]
    plausibles = sejour[sejour["verdict"] == "plausible"]
    eligibles = dans_domaine[dans_domaine["verdict"] == "plausible"]
    print(f"  utilisateurs avec un exposant           : {len(sejour)}")
    print(f"  alpha median (densite)                  : {sejour['alpha'].median():.2f}")
    print(f"  dans le domaine 2 < alpha < 3           : {len(dans_domaine)}  ({len(dans_domaine) / len(sejour):.0%})")
    print(f"  loi de puissance plausible              : {len(plausibles)}  ({len(plausibles) / len(sejour):.0%})")
    print(f"  les deux (seuls comparables a la theorie) : {len(eligibles)}")
    if eligibles.empty:
        print("  -> Aucun utilisateur n'est a la fois dans le domaine et plausible : la prediction")
        print("     H = (3 - a) / 2 ne s'applique pas a ces donnees. C'est un resultat a rapporter.")
        return pd.DataFrame()

    rng = rng or np.random.default_rng(config.GRAINE)
    lignes = []
    for _, r in eligibles.iterrows():
        chemin_h = os.path.join(dossier_resultats_h, f"{r['population']}_H.csv")
        if not os.path.exists(chemin_h):
            continue
        h = pd.read_csv(chemin_h)
        if h.empty or "h_dfa" not in h.columns or h["h_dfa"].notna().sum() == 0:
            continue
        n_serie = int(h["n_points"].median()) - 1
        pas = float(h["pas_median_s"].median()) if "pas_median_s" in h.columns and h["pas_median_s"].notna().any() else 5.0
        alpha_survie = float(r["alpha"]) - 1.0
        xm_en_pas = max(1.0, float(r["xmin"]) / pas)           # duree minimale d'un arret, en nombre de points
        simules = [dfa_h(on_off(n_serie, 3.0, alpha_survie, rng, xm=xm_en_pas)) for _ in range(nb_rep)]
        h_simule = float(np.nanmean(simules))
        h_theorique = h_theorique_on_off(alpha_survie, alpha_survie)
        h_observe = float(h["h_dfa"].median())
        lignes.append(dict(
            utilisateur=r["population"], alpha=float(r["alpha"]), alpha_survie=alpha_survie, xmin_s=float(r["xmin"]),
            p_bootstrap=float(r["p_bootstrap"]), n_serie=n_serie, h_theorique=h_theorique,
            h_simule_meme_N=h_simule, h_observe=h_observe,
            ecart_obs_theorie=h_observe - h_theorique, ecart_obs_simule=h_observe - h_simule,
            part_dfa_sup_1=float(h["dfa_sup_1"].mean()) if "dfa_sup_1" in h.columns else float("nan")))
    resultat = pd.DataFrame(lignes)
    if resultat.empty:
        print("  (aucun fichier resultats_H/<user>_H.csv correspondant : lancez 03_estimation_H.py)")
        return resultat
    print(f"\n  {len(resultat)} utilisateurs comparables. Ecarts a la theorie (H observe - H predit) :")
    print(f"    mediane {resultat['ecart_obs_theorie'].median():+.3f} ; "
          f"a la meme longueur (H observe - H simule) : mediane {resultat['ecart_obs_simule'].median():+.3f}")
    return resultat


def main():
    analyseur = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analyseur.add_argument("--reel", action="store_true", help="partie 2 : donnees reelles (au lieu de la partie 1)")
    analyseur.add_argument("--n", type=int, default=3000, help="partie 1 : longueur des series simulees")
    analyseur.add_argument("--n-rep", type=int, default=60, help="partie 1 : repetitions")
    analyseur.add_argument("--sortie", default=os.path.join("resultats_H", "experiences", "mecanisme_on_off_synthetique.csv"))
    analyseur.add_argument("--lois", default=os.path.join("resultats_H", "lois_de_puissance", "lois_resume.csv"),
                           help="partie 2 : lois_resume.csv de experiences/lois_de_puissance.py")
    analyseur.add_argument("--resultats-h", default="resultats_H", help="partie 2 : dossier des <user>_H.csv")
    analyseur.add_argument("--n-rep-simulation", type=int, default=30, help="partie 2 : simulations calibrees par utilisateur")
    args = analyseur.parse_args()

    if args.reel:
        resultat = partie_reelle(args.lois, args.resultats_h, nb_rep=args.n_rep_simulation)
        if not resultat.empty:
            chemin = os.path.join("resultats_H", "experiences", "mecanisme_on_off_reel.csv")
            os.makedirs(os.path.dirname(chemin), exist_ok=True)
            resultat.to_csv(chemin, index=False)
            print(resultat.round(3).to_string(index=False))
            print(f"\n-> {chemin}")
        return

    df = verifier_theorie(n=args.n, nb_rep=args.n_rep)
    os.makedirs(os.path.dirname(args.sortie), exist_ok=True)
    df.to_csv(args.sortie, index=False)
    print(f"H estime sur ON/OFF a durees de Pareto (N = {args.n}, {args.n_rep} repetitions) :\n")
    colonnes = ["alpha", "h_theorique", "dfa_moyen", "dfa_biais", "rs_corrige_moyen", "rs_corrige_biais", "whittle_moyen", "whittle_biais"]
    print(df[colonnes].round(3).to_string(index=False))
    print("\nPartie 2 (donnees reelles) : python experiences/lois_de_puissance.py, puis ce script avec --reel")


if __name__ == "__main__":
    main()
