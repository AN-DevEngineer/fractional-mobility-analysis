# ./pretraitement/tri.py
#
# Module séparé de nettoyage.py, conformément à la structure cible
# décrite dans reflexion_revisee.pdf (§6 "Structure des fichiers").
# Fonctions inchangées par rapport à l'ancien nettoyage.py — seul leur
# emplacement change, et surtout leur position dans le pipeline
# (cf. main.py) : trier_par_temps() est désormais appelé AVANT
# supprimer_doublons(), et non après.
#
# Pourquoi ce changement d'ordre est important :
# l'ancien pipeline appelait trier_par_temps() puis supprimer_doublons()
# dans cet ordre déjà — mais nettoyer_points() (validation des
# coordonnées) était appelé APRÈS ce tri+déduplication. Si des points
# invalides (lat/lon hors limites, timestamp <= 0) se glissaient dans
# le tri, un timestamp corrompu pouvait fausser l'ordre de toute la
# trajectoire. La nouvelle séquence (cf. main.py) est :
#   nettoyer_points() -> trier_par_temps() -> supprimer_doublons()
# afin que le tri et la déduplication s'appliquent toujours à des
# points déjà valides.


def trier_par_temps(points):
    """
    Trie une liste de points par timestamp croissant

    Entrée :
        points : [(lat, lon, alt, timestamp), ...]

    Sortie :
        list : trié par timestamp
    """
    return sorted(points, key=lambda p: p[3])


def verifier_ordre_temporel(points):
    """
    Vérifie que les timestamps sont strictement croissants
    Utile pour le débogage après trier_par_temps()

    Entrée :
        points : [(lat, lon, alt, timestamp), ...]

    Sortie :
        list of int : indices où le temps recule (liste vide = tout est bon)
    """
    erreurs = []

    for i in range(1, len(points)):
        if points[i][3] < points[i - 1][3]:
            erreurs.append(i)

    return erreurs
