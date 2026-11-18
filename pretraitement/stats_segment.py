# ./pretraitement/stats_segment.py
#
# NOUVEAU MODULE — Phase 2 du pipeline révisé
# (cf. reflexion_revisee.pdf §3 "Trois phases distinctes")
#
# Cette phase calcule, une seule fois par segment, des statistiques
# descriptives globales (médiane de vitesse, écart-type, saut maximal,
# etc.). Ces statistiques servent de contexte à la Phase 3
# (pretraitement/filtres.py) : elles permettent de juger un point par
# rapport au comportement global du segment, et pas seulement par
# rapport à son voisin immédiat — ce qui était la limite de l'ancien
# filtrer_vitesses() (calculs.py, retiré dans cette révision).
#
# Implémentation vectorisée avec numpy/pandas, conformément à la note
# sur les bibliothèques (reflexion_revisee.pdf §8) : chaque opération
# vectorisée est commentée pour expliciter à quelle boucle Python
# explicite elle correspond.
#
# ============================================================
# V5 (cf. documentation/CONSTANTES.md) — modifications apportées :
# ------------------------------------------------------------
# 1. AJOUT de clés contextuelles utilisees par les seuils RELATIFS
#    de filtres.py (etape 3 du plan) :
#       - pas_median_s       : alias explicite de mediane_dt, destine
#                              a journal.py et courbe.py (seuil de gap
#                              relatif : SEUIL_GAP_NOTABLE_RELATIF * pas)
#       - mediane_distance_m : distance_mediane convertie en metres,
#                              destinee a filtrer_saut_distance (seuil
#                              MAX_DIST_M * mediane_distance_m)
#       - mad_vitesse        : MAD des vitesses valides, destine au
#                              seuil MAD_DELTA_V_FACTEUR_MAD * mad
#    Ces cles sont la SEULE source de verite pour les modules aval :
#    filtres.py n'a plus a recalculer une mediane en interne.
#
# 2. STATIONNARITE : le test KS 2-echantillons (scipy.stats.ks_2samp)
#    REMPLACE le ratio heuristique premiere/deuxieme moitie comme
#    critere de DECISION. Nouvelle cle decisionnelle :
#       - statio_non_stationnaire : bool (p < STATIO_ALPHA)
#    et nouvelle cle informative :
#       - statio_p_value          : float (ou None si scipy absent)
#    Le ratio V4 (ratio_stationnarite) est CONSERVE dans le dict
#    pour l'affichage dans les logs (indicateur rapide a l'oeil),
#    mais il ne pilote plus aucune decision.
#    Si scipy est absent, fallback automatique sur le ratio V4
#    (statio_non_stationnaire = ratio hors [MIN, MAX]).
#
# 3. Les cles V4 (mediane_dt, distance_mediane, ratio_heterogeneite,
#    ratio_stationnarite) sont CONSERVEES telles quelles : aucune
#    regression pour journal.py, courbe.py, estimation_segment.py
#    pendant la migration.
# ============================================================

import numpy as np
import pandas as pd

from config import (
    STATIO_TEST, STATIO_ALPHA, STATIO_MIN_POINTS,
    STATIO_RATIO_MIN, STATIO_RATIO_MAX,
)
from pretraitement.geodesie import haversine_vectorise


def segment_en_dataframe(segment):
    """
    Convertit un segment (liste de tuples) en DataFrame pandas

    À partir de la Phase 2, un segment n'est plus transporté comme une
    simple liste de tuples (lat, lon, alt, timestamp) mais comme un
    DataFrame : cela permet d'attacher à chaque point des colonnes
    supplémentaires en Phase 3 (flags "suspect_dist", "suspect_vitesse")
    sans devoir reconstruire des tuples immuables à chaque marquage.

    Entrée :
        segment : liste de tuples (lat, lon, alt, timestamp), triée
                  par temps croissant (sortie de segmenter_points())

    Sortie :
        pandas.DataFrame : colonnes "lat", "lon", "alt", "timestamp"
    """
    return pd.DataFrame(segment, columns=["lat", "lon", "alt", "timestamp"])


def _test_stationnarite_ks(distances):
    """
    Test KS 2-echantillons entre les deux moities de la serie des distances.

    Pourquoi KS plutot que le ratio V4 (moyenne2 / moyenne1) :
    le ratio ne regarde que la MOYENNE et ignore la forme de la
    distribution. Une serie qui passe de "petits pas bruites" a "pas
    reguliers" peut avoir le meme ratio que l'inverse. KS compare les
    FONCTIONS DE REPARTITION empiriques : il detecte aussi les
    changements de forme (variance, asymetrie), pas seulement de
    niveau. C'est le test standard pour "deux echantillons
    proviennent-ils de la meme loi ?" (Massey 1951, Annals of
    Mathematical Statistics, 22(1), 68-77).

    Retour :
        (p_value: float ou None, non_stationnaire: bool ou None)
        (None, None) si scipy absent ou si le segment est trop court
    """
    if STATIO_TEST != "ks":
        return None, None
    if len(distances) < 2 * STATIO_MIN_POINTS:
        return None, None
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        # scipy non installe -> fallback gere par calculer_stats_segment()
        return None, None
    moitie1, moitie2 = np.array_split(distances, 2)
    p = float(ks_2samp(moitie1, moitie2).pvalue)
    return p, bool(p < STATIO_ALPHA)


def calculer_stats_segment(segment):
    """
    Calcule les statistiques descriptives d'un segment (Phase 2)

    Entrée :
        segment : liste de tuples (lat, lon, alt, timestamp), triée
                  par temps croissant (sortie de segmenter_points())

    Sortie :
        dict avec les clés :
            # --- V4 (inchangees) ---
            n_points           : int, nombre de points du segment
            mediane_dt         : médiane des intervalles de temps (s)
            ecart_type_dt      : écart-type des intervalles de temps (s)
            mediane_vitesse    : médiane des vitesses instantanées (km/h)
            ecart_type_vitesse : écart-type des vitesses instantanées (km/h)
            vitesse_max        : vitesse instantanée maximale (km/h)
            distance_mediane   : médiane des distances entre points consécutifs (km)
            distance_max       : distance maximale entre deux points consécutifs (km),
                                  c'est le "saut" le plus grand du segment
            distance_totale    : somme des distances entre points consécutifs (km)
            ratio_heterogeneite: sigma_v / v_mediane (V4, conserve)
            ratio_stationnarite: moyenne moitie2 / moyenne moitie1 (V4, conserve
                                  pour AFFICHAGE uniquement en V5)

            # --- V5 : contexte pour seuils relatifs (filtres.py, journal.py) ---
            pas_median_s       : float, = mediane_dt (alias explicite ; source
                                  unique pour SEUIL_GAP_NOTABLE_RELATIF)
            mediane_distance_m : float, = distance_mediane * 1000 (metres ;
                                  source unique pour MAX_DIST_M relatif)
            mad_vitesse        : float, MAD des vitesses valides (source unique
                                  pour MAD_DELTA_V_FACTEUR_MAD)

            # --- V5 : stationnarite (test KS remplace le ratio en decision) ---
            statio_p_value          : float ou None
            statio_non_stationnaire : bool ou None  (None si test indisponible)

        Toutes les valeurs (sauf n_points) sont None si le segment a
        moins de 2 points (aucun écart calculable).
    """
    n = len(segment)

    stats = {
        # --- V4 ---
        "n_points":              n,
        "mediane_dt":            None,
        "ecart_type_dt":         None,
        "mediane_vitesse":       None,
        "ecart_type_vitesse":    None,
        "vitesse_max":           None,
        "distance_mediane":      None,
        "distance_max":          None,
        "distance_totale":       None,
        "ratio_heterogeneite":   None,
        "ratio_stationnarite":   None,
        # --- V5 (contexte pour filtres relatifs) ---
        "pas_median_s":          None,
        "mediane_distance_m":    None,
        "mad_vitesse":           None,
        # --- V5 (stationnarite) ---
        "statio_p_value":          None,
        "statio_non_stationnaire": None,
    }

    if n < 2:
        return stats

    # --- Conversion en tableau numpy ---
    # arr[:, j] extrait une colonne entière (lat, lon, alt ou t) en un
    # seul tableau 1D. Ceci remplace une boucle Python
    # "for point in segment: lat, lon, alt, t = point" par un accès
    # vectorisé : toutes les latitudes (par exemple) sont obtenues en
    # une seule opération plutôt qu'en itérant point par point.
    arr = np.array(segment, dtype=float)   # shape (n, 4) : lat, lon, alt, t
    lat = arr[:, 0]
    lon = arr[:, 1]
    t   = arr[:, 3]

    # --- delta_t entre points consécutifs ---
    # np.diff(t) calcule en une seule opération tous les
    # t[i] - t[i-1] pour i=1..n-1. C'est l'équivalent vectorisé de la
    # boucle de calculs.calculer_delta_t() de l'ancien pipeline
    # ("for i in range(1, len(points)): delta_t.append(t[i]-t[i-1])").
    delta_t = np.diff(t)

    # --- distances haversine entre points consécutifs ---
    # haversine_vectorise() (geodesie.py) applique la formule de
    # Haversine sur deux tableaux numpy décalés d'un indice
    # (lat[:-1] = tous les points sauf le dernier, lat[1:] = tous les
    # points sauf le premier) : chaque paire (lat[i], lat[i+1]) est
    # ainsi traitée en une seule passe, remplaçant la boucle
    # "for i in range(1, len(points)): haversine(points[i-1], points[i])"
    # de l'ancien calculs.calculer_distance().
    distances = haversine_vectorise(lat[:-1], lon[:-1], lat[1:], lon[1:])

    # --- vitesses instantanées ---
    # Conversion de delta_t en heures, en mettant à NaN les intervalles
    # nuls ou négatifs (désordre temporel résiduel, doublon non
    # détecté) : np.where() évite ici une division par zéro silencieuse
    # qui se serait produite avec une boucle classique sans ce garde-fou.
    delta_t_h = np.where(delta_t > 0, delta_t / 3600.0, np.nan)
    vitesses  = distances / delta_t_h

    # On retire les NaN (cas dt<=0) avant de calculer médiane/écart-type
    # /max, plutôt que de laisser un seul intervalle corrompu fausser
    # toute la statistique du segment.
    vitesses_valides = vitesses[~np.isnan(vitesses)]

    # --- V4 : statistiques descriptives ---
    stats["mediane_dt"]       = float(np.median(delta_t))
    stats["ecart_type_dt"]    = float(np.std(delta_t))
    stats["distance_mediane"] = float(np.median(distances))
    stats["distance_max"]     = float(np.max(distances))
    stats["distance_totale"]  = float(np.sum(distances))

    # --- V5 : contexte pour les seuils relatifs ---
    # Alias explicite de mediane_dt. Meme valeur, mais le nom dit
    # ce que journal.py et courbe.py cherchent (le "pas" de
    # l'echantillonnage), sans avoir a deviner que "mediane_dt" est
    # la bonne cle a utiliser.
    stats["pas_median_s"]       = stats["mediane_dt"]
    # Distance mediane en metres : evite la conversion km->m dans
    # filtres.py (seuil MAX_DIST_M relatif).
    stats["mediane_distance_m"] = stats["distance_mediane"] * 1000.0

    if len(vitesses_valides) > 0:
        stats["mediane_vitesse"]    = float(np.median(vitesses_valides))
        stats["ecart_type_vitesse"] = float(np.std(vitesses_valides))
        stats["vitesse_max"]        = float(np.max(vitesses_valides))

        v_med = float(np.median(vitesses_valides))

        # --- V4 : Ratio R = sigma_v / v_mediane ---
        # Calculé uniquement si la médiane est non nulle pour éviter
        # une division par zéro sur un segment complètement immobile.
        # np.median() est robuste aux valeurs extrêmes, c'est pourquoi
        # on l'utilise comme dénominateur plutôt que la moyenne.
        if v_med > 0:
            stats["ratio_heterogeneite"] = float(np.std(vitesses_valides) / v_med)

        # --- V5 : MAD des vitesses ---
        # MAD = mediane(|v_i - v_med|). Meme calcul que dans
        # filtres.filtrer_vitesse() pour la detection MAD, expose ici
        # pour que filtres.py puisse utiliser MAD_DELTA_V_FACTEUR_MAD
        # sans recalculer le MAD en interne (source unique de verite).
        stats["mad_vitesse"] = float(np.median(np.abs(vitesses_valides - v_med)))

    # --- V4 : Ratio de stationnarité (première moitié / deuxième moitié) ---
    # CONSERVE pour l'AFFICHAGE dans les logs, mais ne pilote plus la
    # decision en V5 (voir statio_non_stationnaire ci-dessous).
    # On utilise les distances (pas les vitesses) pour ne pas dépendre
    # des delta_t, qui peuvent être irréguliers : les distances seules
    # reflètent directement la série analysée par DFA/RS/Whittle.
    if len(distances) >= 4:
        moitie1, moitie2 = np.array_split(distances, 2)
        m1 = float(np.mean(moitie1))
        m2 = float(np.mean(moitie2))
        if m1 > 0:
            stats["ratio_stationnarite"] = round(m2 / m1, 3)

    # --- V5 : test KS 2-echantillons (decision de stationnarite) ---
    # Le test KS compare les fonctions de repartition empiriques des
    # deux moities. Il detecte aussi les changements de FORME
    # (variance, asymetrie) que le ratio de moyennes ignore.
    # Si STATIO_TEST != "ks" ou scipy absent, fallback sur le ratio V4.
    p_value, non_stationnaire = _test_stationnarite_ks(distances)

    if p_value is not None:
        # Test KS disponible : c'est lui qui decide.
        stats["statio_p_value"]          = p_value
        stats["statio_non_stationnaire"] = non_stationnaire
    elif STATIO_TEST == "ratio" or (
            # Fallback : test KS demande mais indisponible (scipy absent
            # ou segment trop court). On utilise le ratio V4 comme
            # approximation, en le rendant EXPLICITEMENT decisionnel.
            STATIO_TEST == "ks" and stats["ratio_stationnarite"] is not None):
        r = stats["ratio_stationnarite"]
        stats["statio_p_value"]          = None    # pas de p-value sous ratio
        stats["statio_non_stationnaire"] = not (STATIO_RATIO_MIN <= r <= STATIO_RATIO_MAX)
    # Si STATIO_TEST == "none" : les deux cles restent None (pas de test).

    return stats