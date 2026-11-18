# ./lois_de_puissance/analyse.py
#
# ANALYSE COMPLETE D'UNE DISTRIBUTION : UNE LIGNE DE RESULTATS PAR VARIABLE
#
# Assemble clauset.py (ajustement et adequation) et alternatives.py (concurrence des lois) et rend un
# dictionnaire a plat, directement exploitable comme ligne de CSV.
#
# ... (en-tete inchange, voir fichier original) ...

import json

import numpy as np

from config import LP_NB_BOOTSTRAP, LP_NB_CANDIDATS_XMIN, LP_NB_MIN_QUEUE, LP_SEUIL_P
# ===== AJOUT 2a : import du test d'adequation de la tronquee =====
from lois_de_puissance.alternatives import (
    NOMS_ALTERNATIVES, NOMS_MODELES, comparer_modeles,
    p_valeur_adequation_tronquee,
)
# ===== FIN AJOUT 2a =====
from lois_de_puissance.clauset import ajuster_loi_de_puissance, p_valeur_adequation

NAN = float("nan")


def _resultat_vide(n, verdict="insuffisant"):
    """Ligne de resultats sans ajustement : toutes les colonnes existent (CSV rectangulaire)."""
    ligne = dict(n=int(n), xmin=NAN, alpha=NAN, sigma_alpha=NAN, n_queue=0, part_queue=NAN, ks=NAN,
                 p_bootstrap=NAN, nb_bootstrap=0, exposant_survie=NAN, dans_domaine_on_off=False,
                 h_on_off_predit=NAN, verdict=verdict, alternatives_meilleures="", meilleur_modele="")
    for nom in NOMS_MODELES:
        ligne[f"poids_aicc_{nom}"] = NAN
    for nom in NOMS_ALTERNATIVES:
        ligne[f"R_{nom}"] = NAN
        ligne[f"p_{nom}"] = NAN
        ligne[f"favorise_{nom}"] = ""
    ligne["parametres_alternatives"] = ""
    # ===== AJOUT 2b : initialisation de la colonne d'adequation tronquee =====
    ligne["p_adequation_tronquee"] = NAN
    # ===== FIN AJOUT 2b =====
    return ligne


def analyser_distribution(x, rng, nb_bootstrap=LP_NB_BOOTSTRAP, nb_candidats=LP_NB_CANDIDATS_XMIN,
                          nb_min_queue=LP_NB_MIN_QUEUE, seuil_p=LP_SEUIL_P,
                          avec_alternatives=True,
                          # ===== AJOUT 2c : nouveaux parametres =====
                          adequation_tronquee=False, nb_bootstrap_tronquee=200):
                          # ===== FIN AJOUT 2c =====
    """
    Ajuste une loi de puissance a la queue de x, teste son adequation et la compare aux alternatives.

    Entree :
        x                    : observations (durees de sejour en s, distances en m...) ; les valeurs
                               non positives ou non finies sont ignorees
        rng                  : numpy.random.Generator (bootstrap : reproductibilite)
        nb_bootstrap         : simulations de la p-valeur d'adequation (0 = ne pas la calculer)
        avec_alternatives    : comparer aux quatre alternatives (plus lent : optimisations numeriques)
        adequation_tronquee  : tester l'adequation de la loi tronquee (bootstrap, AJOUT)
        nb_bootstrap_tronquee: simulations pour ce test (defaut 200)

    Sortie : dict a plat
        ... (champs existants inchange) ...
        parametres_alternatives   JSON des parametres ajustes de chaque alternative
        p_adequation_tronquee     p-valeur d'adequation de la tronquee (AJOUT) ;
                                  nan si adequation_tronquee=False ou si la tronquee n'a pas pu
                                  etre ajustee.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    ajustement = ajuster_loi_de_puissance(x, nb_candidats, nb_min_queue)
    if ajustement is None:
        return _resultat_vide(len(x))

    ligne = _resultat_vide(len(x), verdict="indetermine")
    alpha = ajustement.alpha
    domaine = bool(2.0 < alpha < 3.0)
    ligne.update(xmin=ajustement.xmin, alpha=alpha, sigma_alpha=ajustement.sigma_alpha,
                 n_queue=ajustement.n_queue, part_queue=ajustement.n_queue / len(x), ks=ajustement.ks,
                 exposant_survie=alpha - 1.0, dans_domaine_on_off=domaine,
                 h_on_off_predit=(4.0 - alpha) / 2.0 if domaine else NAN)

    if nb_bootstrap > 0:
        ligne["p_bootstrap"] = p_valeur_adequation(x, ajustement, rng, nb_bootstrap, nb_candidats, nb_min_queue)
        ligne["nb_bootstrap"] = int(nb_bootstrap)

    meilleures = []
    if avec_alternatives:
        queue = x[x >= ajustement.xmin]
        comparaison = comparer_modeles(queue, ajustement.xmin, alpha, seuil_p)
        for nom, poids in comparaison["poids_aicc"].items():
            ligne[f"poids_aicc_{nom}"] = poids
        for nom, c in comparaison["comparaisons"].items():
            ligne[f"R_{nom}"] = c["R"]
            ligne[f"p_{nom}"] = c["p"]
            ligne[f"favorise_{nom}"] = c["favorise"]
            if c["favorise"] not in ("loi_de_puissance", "indecidable"):
                meilleures.append(nom)
        ligne["meilleur_modele"] = comparaison["meilleur_modele"] or ""
        ligne["alternatives_meilleures"] = "+".join(meilleures)
        ligne["parametres_alternatives"] = json.dumps(
            comparaison["parametres"], default=str, ensure_ascii=False)
        # ===== AJOUT 2d : test d'adequation de la tronquee =====
        if adequation_tronquee:
            params_tronq = comparaison["parametres"].get("puissance_tronquee", {})
            if "alpha" in params_tronq and "lambda" in params_tronq:
                ligne["p_adequation_tronquee"] = p_valeur_adequation_tronquee(
                    queue,
                    params_tronq["alpha"],
                    params_tronq["lambda"],
                    ajustement.xmin,
                    rng,
                    nb_bootstrap_tronquee,
                )
        # ===== FIN AJOUT 2d =====

    p = ligne["p_bootstrap"]
    if np.isfinite(p):
        if p <= seuil_p:
            ligne["verdict"] = "rejetee"
        elif meilleures:
            ligne["verdict"] = "alternative_meilleure"
        else:
            ligne["verdict"] = "plausible"
    return ligne