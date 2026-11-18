# ./pretraitement/geodesie.py
#
# Calculs de base sur les points GPS : distance de Haversine, vitesse, ecart de temps.
# (Ex-utils.py : la conversion de timestamp a ete deplacee dans lecture.py.)

import math

import numpy as np




def haversine(lat1, lon1, lat2, lon2):
    """
    Calcule la distance entre deux points GPS (en kilomètres)
    Formule de Haversine

    Entrée :
        lat1, lon1 : coordonnées point 1 (degrés décimaux)
        lat2, lon2 : coordonnées point 2 (degrés décimaux)

    Sortie :
        float : distance en km
    """
    R = 6371  # rayon moyen de la Terre en km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    d_phi    = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def difference_temps(t1, t2):
    """
    Calcule l'écart entre deux timestamps

    Entrée :
        t1, t2 : timestamps en secondes

    Sortie :
        float : delta_t en secondes
    """
    return t2 - t1


# ------------------------------------------------------------
# Versions vectorisées (numpy) — utilisées en Phase 2 et Phase 3
# du pipeline révisé (stats_segment.py, filtres.py).
#
# Note (cf. reflexion_revisee.pdf §8 "Bibliothèques") : numpy/pandas
# sont désormais autorisés, à condition de documenter chaque opération
# vectorisée avec la même précision qu'une boucle explicite. C'est
# l'objet des commentaires ci-dessous : chaque ligne numpy remplace
# une boucle Python "for i in range(...)" sans en changer le résultat.
# ------------------------------------------------------------

def haversine_vectorise(lat1, lon1, lat2, lon2):
    """
    Version vectorisée de haversine() : accepte des tableaux numpy
    (ou des scalaires) pour lat1/lon1/lat2/lon2 et renvoie toutes les
    distances en une seule opération, au lieu d'appeler haversine()
    point par point dans une boucle Python.

    Correspondance terme à terme avec haversine() ci-dessus :
        - np.radians(x)        remplace math.radians(x) appliqué à
                                un tableau entier plutôt qu'à un float
        - np.sin / np.cos      remplacent math.sin / math.cos,
                                appliqués élément par élément
        - np.sqrt / np.arctan2 remplacent math.sqrt / math.atan2,
                                également élément par élément
    Aucune formule n'est changée : seule la boucle Python implicite
    devient une opération numpy vectorisée (calcul en une passe sur
    tout le tableau au lieu d'une itération point par point).

    Entrée :
        lat1, lon1, lat2, lon2 : float ou numpy.ndarray, degrés décimaux
                                  (doivent avoir la même forme si ce
                                  sont des tableaux)

    Sortie :
        float ou numpy.ndarray : distance(s) en km, même forme que
                                  l'entrée
    """
    R = 6371  # rayon moyen de la Terre en km

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    d_phi    = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)

    a = (np.sin(d_phi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def vitesse_kmh(distance_km, delta_t_s):
    """
    Calcule une vitesse en km/h à partir d'une distance (km) et d'un
    intervalle de temps (s).

    Fonction scalaire simple, nommée ainsi pour correspondre à
    l'interface évoquée dans les documents de réflexion pour
    pretraitement/filtres.py (utile pour des vérifications ponctuelles
    ou des tests ; les calculs en masse sur un segment entier passent
    par les opérations numpy vectorisées de stats_segment.py et
    filtres.py, qui ne rappellent pas cette fonction point par point).

    Entrée :
        distance_km : float
        delta_t_s   : float, en secondes

    Sortie :
        float ou None : vitesse en km/h, None si delta_t_s <= 0
                         (intervalle nul, négatif, ou doublon résiduel)
    """
    if delta_t_s <= 0:
        return None

    return distance_km / (delta_t_s / 3600.0)
