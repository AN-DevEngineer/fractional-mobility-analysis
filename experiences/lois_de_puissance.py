# ./experiences/lois_de_puissance.py
# ... (en-tete inchange) ...

import argparse
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from lois_de_puissance.analyse import analyser_distribution
from pretraitement.arrets import (distances_entre_arrets, lire_points_sortie, sejours_utilisateur,
                                  temps_de_sejour)


def nom_jeu(rayon_m, duree_min_s):
    """Nom lisible d'un jeu de seuils, ex. (100.0, 300.0) -> '100m_300s'."""
    return f"{rayon_m:g}m_{duree_min_s:g}s"


def jeu_par_defaut():
    """Jeu de seuils de config.py (ARRET_RAYON_M, ARRET_DUREE_MIN_S), au format de nom_jeu()."""
    return nom_jeu(config.ARRET_RAYON_M, config.ARRET_DUREE_MIN_S)


def collecter_sejours(dossier_sortie, jeux, utilisateurs=None):
    """... (inchange) ..."""
    parties = []
    for nom in sorted(os.listdir(dossier_sortie)):
        if not (nom.endswith(".csv") and nom[:-4].isdigit()):
            continue
        utilisateur = nom[:-4]
        if utilisateurs and utilisateur not in utilisateurs:
            continue
        points = lire_points_sortie(os.path.join(dossier_sortie, nom))
        for rayon, duree in jeux:
            s = sejours_utilisateur(points, utilisateur, rayon_m=rayon, duree_min_s=duree)
            s["jeu"] = nom_jeu(rayon, duree)
            parties.append(s)
    if not parties:
        return pd.DataFrame()
    return pd.concat(parties, ignore_index=True)


def _graine(cle):
    """Generateur propre a une analyse : le resultat ne depend que de (GRAINE, cle), pas de l'ordre."""
    return np.random.default_rng(np.random.SeedSequence([config.GRAINE, zlib.crc32(cle.encode("utf-8"))]))


def _valeurs(sejours, variable):
    """... (inchange) ..."""
    if variable == "sejour":
        return temps_de_sejour(sejours, seulement_non_censurees=True)
    if variable == "sejour_avec_censurees":
        return temps_de_sejour(sejours, seulement_non_censurees=False)
    if variable == "distance":
        return distances_entre_arrets(sejours, meme_segment_seulement=True)["distance_m"].to_numpy(dtype=float)
    if variable == "distance_a_cheval":
        return distances_entre_arrets(sejours, meme_segment_seulement=False)["distance_m"].to_numpy(dtype=float)
    raise ValueError(f"variable inconnue : {variable!r}")


def resumer_distributions(sejours, nb_bootstrap, nb_bootstrap_utilisateur,
                          n_min_utilisateur=config.LP_NB_MIN_PAR_UTILISATEUR, un_seul_jeu=False,
                          avec_alternatives=True,
                          # ===== AJOUT 3a : nouveau parametre =====
                          adequation_tronquee=False):
                          # ===== FIN AJOUT 3a =====
    """
    Analyse chaque (jeu, population, variable) et renvoie un DataFrame d'une ligne par analyse.

    Entree :
        sejours                  : DataFrame de collecter_sejours()
        nb_bootstrap             : simulations pour la population "agrege"
        nb_bootstrap_utilisateur : simulations pour chaque utilisateur
        n_min_utilisateur        : observations minimales pour analyser un utilisateur isole
        un_seul_jeu              : ne traiter que le jeu par defaut
        avec_alternatives        : comparer aux quatre alternatives (lent)
        adequation_tronquee      : tester l'adequation de la tronquee (AJOUT, lent)
    """
    if len(sejours) == 0:
        return pd.DataFrame()
    defaut = jeu_par_defaut()
    lignes = []
    for jeu, s_jeu in sejours.groupby("jeu", sort=False):
        if un_seul_jeu and jeu != defaut:
            continue
        variables = ["sejour", "distance"]
        if jeu == defaut:
            variables += ["sejour_avec_censurees", "distance_a_cheval"]
        populations = [("agrege", s_jeu, nb_bootstrap)]
        populations += [(u, s_u, nb_bootstrap_utilisateur) for u, s_u in s_jeu.groupby("utilisateur", sort=True)]
        for population, s_pop, nb_boot in populations:
            for variable in variables:
                x = _valeurs(s_pop, variable)
                if population != "agrege" and len(x) < n_min_utilisateur:
                    continue
                resultat = analyser_distribution(
                    x,
                    _graine(f"{jeu}|{population}|{variable}"),
                    nb_bootstrap=nb_boot,
                    avec_alternatives=avec_alternatives,
                    # ===== AJOUT 3b : propagation du parametre =====
                    adequation_tronquee=adequation_tronquee,
                    # ===== FIN AJOUT 3b =====
                )
                lignes.append(dict(jeu=jeu, population=population, variable=variable,
                                   n_observations=int(len(x)), **resultat,
                                   empreinte_config=config.empreinte_config()))
    return pd.DataFrame(lignes)


def afficher_resume(resume):
    """... (inchange) ..."""
    agrege = resume[resume["population"] == "agrege"]
    if agrege.empty:
        print("Aucune analyse sur la population agregee.")
        return
    colonnes = ["jeu", "variable", "n_observations", "n_queue", "xmin", "alpha", "p_bootstrap",
                "verdict", "meilleur_modele", "dans_domaine_on_off"]
    # AJOUT : on inclut p_adequation_tronquee si la colonne existe et a des valeurs
    if "p_adequation_tronquee" in agrege.columns and agrege["p_adequation_tronquee"].notna().any():
        colonnes.append("p_adequation_tronquee")
    print("\nPopulation agregee (tous les utilisateurs) :")
    print(agrege[colonnes].round(3).to_string(index=False))
    par_utilisateur = resume[(resume["population"] != "agrege") & (resume["jeu"] == jeu_par_defaut())
                             & (resume["variable"] == "sejour")]
    if not par_utilisateur.empty:
        print(f"\nPar utilisateur (jeu {jeu_par_defaut()}, variable sejour) : {len(par_utilisateur)} utilisateurs analyses")
        print(par_utilisateur["verdict"].value_counts().to_string())
        print(f"alpha median {par_utilisateur['alpha'].median():.2f} ; "
              f"part dans le domaine 2 < alpha < 3 du lien ON/OFF : {par_utilisateur['dans_domaine_on_off'].mean():.0%}")


def main():
    analyseur = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analyseur.add_argument("--sortie", default="sortie", help="dossier des CSV de pretraitement")
    analyseur.add_argument("--resultats", default=os.path.join("resultats_H", "lois_de_puissance"))
    analyseur.add_argument("--utilisateurs", nargs="*", default=None, help="restreindre (ex. 000 001)")
    analyseur.add_argument("--n-bootstrap", type=int, default=config.LP_NB_BOOTSTRAP,
                           help="simulations pour la population agregee")
    analyseur.add_argument("--n-bootstrap-utilisateur", type=int, default=50,
                           help="simulations par utilisateur (0 = sans p-valeur)")
    analyseur.add_argument("--un-seul-jeu", action="store_true",
                           help="ne traiter que le jeu de seuils par defaut (plus rapide)")
    analyseur.add_argument("--sans-alternatives", action="store_true",
                           help="ne pas comparer aux lois alternatives (plus rapide)")
    # ===== AJOUT 3c : flag CLI =====
    analyseur.add_argument("--adequation-tronquee", action="store_true",
                           help="tester l'adequation de la tronquee (bootstrap, lent)")
    # ===== FIN AJOUT 3c =====
    args = analyseur.parse_args()

    jeux = [(config.ARRET_RAYON_M, config.ARRET_DUREE_MIN_S)] if args.un_seul_jeu else list(config.ARRET_JEUX_SENSIBILITE)
    if (config.ARRET_RAYON_M, config.ARRET_DUREE_MIN_S) not in jeux:
        jeux.insert(0, (config.ARRET_RAYON_M, config.ARRET_DUREE_MIN_S))

    if not os.path.isdir(args.sortie):
        print(f"[ERREUR] Dossier introuvable : {args.sortie}")
        return
    print(f"Detection des arrets pour {len(jeux)} jeu(x) de seuils : {[nom_jeu(*j) for j in jeux]}")
    sejours = collecter_sejours(args.sortie, jeux, args.utilisateurs)
    if sejours.empty:
        print("Aucun sejour detecte (dossier vide, ou aucun segment assez long).")
        return
    print(f"{len(sejours)} sejours (toutes lignes, tous jeux) pour {sejours['utilisateur'].nunique()} utilisateur(s)")

    os.makedirs(args.resultats, exist_ok=True)
    sejours.to_csv(os.path.join(args.resultats, "sejours.csv"), index=False)
    print("Ajustement des lois de puissance (le bootstrap peut etre long)...")
    resume = resumer_distributions(
        sejours, args.n_bootstrap, args.n_bootstrap_utilisateur,
        un_seul_jeu=args.un_seul_jeu,
        avec_alternatives=not args.sans_alternatives,
        # ===== AJOUT 3d : propagation du flag =====
        adequation_tronquee=args.adequation_tronquee,
        # ===== FIN AJOUT 3d =====
    )
    resume.to_csv(os.path.join(args.resultats, "lois_resume.csv"), index=False)
    afficher_resume(resume)
    print(f"\nResultats dans {args.resultats}/ (sejours.csv, lois_resume.csv)")


if __name__ == "__main__":
    main()