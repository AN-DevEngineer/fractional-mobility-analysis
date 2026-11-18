# ./pretraitement/nettoyage.py

def point_valide(point):
    """
    Vérifie si un point GPS est valide

    Entrée :
        point : (lat, lon, alt, timestamp)

    Sortie :
        bool : True si le point est utilisable
    """
    lat, lon, alt, t = point

    if lat < -90 or lat > 90:
        return False

    if lon < -180 or lon > 180:
        return False

    # -777 est la valeur sentinelle GeoLife pour altitude inconnue
    # if alt == -777:
    #     return False

    if t <= 0:
        return False

    return True


def nettoyer_points(points):
    """
    Supprime les points invalides d'une liste

    Entrée :
        points : [(lat, lon, alt, timestamp), ...]

    Sortie :
        list : points valides uniquement
    """
    points_propres = []

    for point in points:
        if point_valide(point):
            points_propres.append(point)

    return points_propres


def supprimer_doublons(points):
    """
    Supprime les doublons consécutifs (même position ET même timestamp)

    Entrée :
        points : [(lat, lon, alt, timestamp), ...]

    Sortie :
        list : sans doublons consécutifs
    """
    if len(points) == 0:
        return []

    points_uniques = [points[0]]

    for i in range(1, len(points)):
        prev = points_uniques[-1]
        curr = points[i]

        # Même timestamp = doublon, quelle que soit la position
        # (jitter GPS : coordonnées légèrement différentes, même instant)
        if curr[3] == prev[3]:
            continue

        points_uniques.append(curr)

    return points_uniques
