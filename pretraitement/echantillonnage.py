# ./pretraitement/echantillonnage.py
#
# Diagnostic du MODE D'ENREGISTREMENT d'un fichier GPS.
#
# GeoLife melange deux modes (guide officiel du jeu de donnees) : un point toutes les 1 a 5 secondes
# (mode TEMPS), ou un point tous les 5 a 10 metres (mode DISTANCE). En mode distance, d_i est quasi
# constant quelle que soit la vitesse : la serie des d_i decrit alors le logger, pas le deplacement.
# Ce diagnostic permet de le detecter, de stratifier les analyses et de mesurer l'effet du mode sur H
# (hypothese H3 du memoire).
#
# Heuristique (seuils dans config.py, section ECHANT_*, a valider sur le corpus complet avec
# 00_audit_echantillonnage.py) :
#   temps     si une part >= ECHANT_PART_TEMPS des pas est a +/- ECHANT_TOL_DT du pas median
#   distance  sinon, si une part >= ECHANT_PART_DISTANCE des distances (pas en mouvement) est a
#             +/- ECHANT_TOL_D de la mediane
#   mixte     dans les autres cas

import numpy as np

from config import (ECHANT_TOL_DT, ECHANT_TOL_D, ECHANT_PART_TEMPS, ECHANT_PART_DISTANCE,
                    ECHANT_DEPLACEMENT_MIN_M)


def profil_echantillonnage(dt, d):
    """
    Entree :
        dt : ecarts de temps entre points successifs (secondes)
        d  : distances entre points successifs (metres), meme longueur que dt

    Sortie : dict
        mode                  "temps" | "distance" | "mixte" | "indetermine"
        pas_median            mediane de dt (s)
        cv_dt                 ecart-type / moyenne de dt
        part_dt_reguliers     part des pas a +/- ECHANT_TOL_DT du pas median
        distance_mediane_mvt  mediane de d sur les pas en mouvement (m)
        part_d_regulieres     part des distances a +/- ECHANT_TOL_D de cette mediane
        corr_d_dt             correlation entre d_i et dt_i (proche de 1 : d_i ne mesure que dt_i)
    """
    dt = np.asarray(dt, float)
    d = np.asarray(d, float)
    nan = float("nan")
    if len(dt) < 2:
        return dict(mode="indetermine", pas_median=nan, cv_dt=nan, part_dt_reguliers=nan,
                    distance_mediane_mvt=nan, part_d_regulieres=nan, corr_d_dt=nan)
    pas_med = float(np.median(dt))
    part_dt = float(np.mean(np.abs(dt - pas_med) <= ECHANT_TOL_DT * pas_med))
    mouvement = d >= ECHANT_DEPLACEMENT_MIN_M
    if mouvement.sum() >= 10:
        d_med = float(np.median(d[mouvement]))
        part_d = float(np.mean(np.abs(d[mouvement] - d_med) <= ECHANT_TOL_D * d_med))
    else:
        d_med, part_d = nan, nan
    if part_dt >= ECHANT_PART_TEMPS:
        mode = "temps"
    elif np.isfinite(part_d) and part_d >= ECHANT_PART_DISTANCE:
        mode = "distance"
    else:
        mode = "mixte"
    corr = float(np.corrcoef(d, dt)[0, 1]) if dt.std() > 0 and d.std() > 0 else nan
    return dict(mode=mode, pas_median=pas_med, cv_dt=float(dt.std() / dt.mean()), part_dt_reguliers=part_dt,
                distance_mediane_mvt=d_med, part_d_regulieres=part_d, corr_d_dt=corr)
