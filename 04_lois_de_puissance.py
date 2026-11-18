#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_lois_de_puissance.py — point d'entree racine pour les lois de puissance.

Trois modes :

  (defaut)   Analyse RAPIDE sur sortie/ : 1 jeu de seuils, 50 bootstraps.
             But : voir un premier resultat en 2-3 min.

  --complet  Analyse COMPLETE : tous les jeux de seuils, 1000 bootstraps,
             avec alternatives. Lent (10-30 min).

  --demo     Validation sur donnees SYNTHETIQUES.

Options :
  --avec-alternatives   comparer aux 4 lois alternatives (lent) ;
                        compatible avec le mode rapide.

Usage :
  python3.13 04_lois_de_puissance.py                          # rapide
  python3.13 04_lois_de_puissance.py --avec-alternatives --n-bootstrap 500
  python3.13 04_lois_de_puissance.py --complet
  python3.13 04_lois_de_puissance.py --demo
"""

import argparse
import os
import sys

import numpy as np

try:
    from config import ARRET_JEUX_SENSIBILITE, ARRET_RAYON_M, ARRET_DUREE_MIN_S
    from experiences.lois_de_puissance import (
        afficher_resume,
        collecter_sejours,
        nom_jeu,
        resumer_distributions,
    )
    from lois_de_puissance.analyse import analyser_distribution
    from lois_de_puissance.clauset import simuler_loi_de_puissance
except ImportError as e:
    print(f"[ERREUR] Import impossible : {e}")
    print("Verifiez les constantes ARRET_* et LP_NB_MIN_PAR_UTILISATEUR dans config.py.")
    sys.exit(1)


def _fmt(val):
    try:
        return f"{val:.3f}" if np.isfinite(val) else "nan"
    except (TypeError, ValueError):
        return "nan"


def _demo():
    print("=" * 70)
    print("DEMO SYNTHETIQUE — verification du pipeline")
    print("=" * 70)
    rng = np.random.default_rng(42)

    x = simuler_loi_de_puissance(1000, 2.5, 1.0, rng)
    r = analyser_distribution(x, rng, nb_bootstrap=100, avec_alternatives=True)
    print("Cas 1 — loi de puissance (alpha_vrai = 2.5)")
    print(f"  alpha        = {_fmt(r['alpha'])}       (attendu ~2.5)")
    print(f"  p_bootstrap  = {_fmt(r['p_bootstrap'])}       (attendu > 0.1)")
    print(f"  verdict      = {r['verdict']}")
    print(f"  meilleur     = {r['meilleur_modele']}")
    print(f"  h_on_off     = {_fmt(r['h_on_off_predit'])}")
    print()

    x = 1.0 + rng.exponential(1.0, 1000)
    r = analyser_distribution(x, rng, nb_bootstrap=100, avec_alternatives=True)
    print("Cas 2 — exponentielle (PAS une loi de puissance)")
    print(f"  p_bootstrap  = {_fmt(r['p_bootstrap'])}       (attendu petit)")
    print(f"  verdict      = {r['verdict']}")
    print(f"  meilleur     = {r['meilleur_modele']}       (attendu != loi_de_puissance)")


def _reel(args):
    if not os.path.isdir(args.sortie):
        print(f"[ERREUR] dossier {args.sortie}/ introuvable.")
        return 1

    fichiers = [f for f in os.listdir(args.sortie) if f.endswith(".csv") and f[:3].isdigit()]
    if not fichiers:
        print(f"[ERREUR] aucun fichier <user>.csv dans {args.sortie}/.")
        return 1
    print(f"Fichiers dans {args.sortie}/ : {len(fichiers)}")
    
    rayon = args.arret_rayon if args.arret_rayon is not None else ARRET_RAYON_M
    duree = args.arret_duree if args.arret_duree is not None else ARRET_DUREE_MIN_S

    if args.tous_les_jeux:
        jeux = list(ARRET_JEUX_SENSIBILITE)
        n_boot = 50 if args.n_bootstrap is None else args.n_bootstrap
        n_boot_user = 0
        avec_alternatives = False
        # adequation_tronquee force a False si tous_les_jeux (test couteux)
        args.adequation_tronquee = False
        un_seul_jeu = False
    elif args.complet:
        jeux = list(ARRET_JEUX_SENSIBILITE)
        if (ARRET_RAYON_M, ARRET_DUREE_MIN_S) not in jeux:
            jeux.insert(0, (ARRET_RAYON_M, ARRET_DUREE_MIN_S))
        n_boot = 1000 if args.n_bootstrap is None else args.n_bootstrap
        n_boot_user = 50
        avec_alternatives = True
        un_seul_jeu = False
    else:
        jeux = [(rayon, duree)]
        n_boot = 50 if args.n_bootstrap is None else args.n_bootstrap
        n_boot_user = 0
        avec_alternatives = args.avec_alternatives   # <-- MODIF
        un_seul_jeu = False

    print(f"Mode                : {'complet' if args.complet else 'rapide'}")
    print(f"Jeux de seuils      : {[nom_jeu(*j) for j in jeux]}")
    print(f"Bootstrap agrege    : {n_boot}")
    print(f"Bootstrap/user      : {n_boot_user}")
    print(f"Alternatives        : {avec_alternatives}")
    print()

    print("Detection des arrets...")
    sejours = collecter_sejours(args.sortie, jeux, args.utilisateurs)
    if sejours.empty:
        print("[ERREUR] aucun sejour detecte.")
        return 1
    print(f"Sejours detectes    : {len(sejours)}")
    print(f"Utilisateurs        : {sejours['utilisateur'].nunique()}")
    
    mode = "complet" if args.complet else ("tous_jeux" if args.tous_les_jeux else "rapide")
    suffixe = "avec_alt" if avec_alternatives else "sans_alt"
    
    if args.sortie != "sortie":
        mode = f"{mode}_{os.path.basename(args.sortie.rstrip('/'))}"

    dossier = os.path.join("resultats_H", "lois_de_puissance")
    os.makedirs(dossier, exist_ok=True)
    nom_sejour = f"sejour_{mode}_{suffixe}_B{n_boot}.csv" 
    
    sejours.to_csv(os.path.join(dossier, nom_sejour), index=False)
    

    print()
    print("Ajustement des lois de puissance...")
    if n_boot >= 200:
        print("(le bootstrap peut prendre plusieurs minutes)")
    resume = resumer_distributions(
        sejours, n_boot, n_boot_user,
        un_seul_jeu=un_seul_jeu,
        avec_alternatives=avec_alternatives,
        adequation_tronquee=args.adequation_tronquee,   # <-- AJOUT
    )
    
    chemin_resume = os.path.join(dossier, f"lois_resume_{mode}_{suffixe}_B{n_boot}.csv")
    
    resume.to_csv(chemin_resume, index=False)
    print()
    afficher_resume(resume)
    print()
    print(f"Resultats : {chemin_resume}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--demo", action="store_true",
                        help="demo synthetique")
    parser.add_argument("--complet", action="store_true",
                        help="analyse complete (lent)")
    parser.add_argument("--avec-alternatives", action="store_true",
                        help="comparer aux 4 lois alternatives (lent)")  # <-- MODIF
    parser.add_argument("--utilisateurs", nargs="*", default=None,
                        help="restreindre a certains utilisateurs")
    parser.add_argument("--n-bootstrap", type=int, default=None,
                        help="nombre de simulations bootstrap")
    parser.add_argument("--tous-les-jeux", action="store_true",
                    help="analyser les 4 jeux de ARRET_JEUX_SENSIBILITE, sans alternatives")
    parser.add_argument("--adequation-tronquee", action="store_true",
                        help="tester l'adequation de la tronquee (bootstrap, lent)")
    parser.add_argument("--sortie", default="sortie",
                    help="dossier des CSV de pretraitement (defaut: sortie/)")
    parser.add_argument("--arret-rayon", type=float, default=None,
                    help="override ARRET_RAYON_M (m)")
    parser.add_argument("--arret-duree", type=float, default=None,
                    help="override ARRET_DUREE_MIN_S (s)")
    args = parser.parse_args()

    if args.demo:
        _demo()
        return 0
    return _reel(args)


if __name__ == "__main__":
    raise SystemExit(main())