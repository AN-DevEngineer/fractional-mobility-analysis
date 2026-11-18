# ./hurst/serie_distances.py
#
# Serie des distances inter-points et profil cumule : entree commune aux trois
# methodes d'estimation de Hurst (DFA, R/S, Whittle). (Ex-moitie de stats_segment_gps.py)

from pretraitement.geodesie import haversine


def extraction_distances(points_ou_serie):
    """
    Extrait les distances inter-points en metres

    Accepte deux formats d'entree :
        - Liste de points GPS : [[lat, lon, timestamp], ...]
          -> calcule les distances haversine entre points successifs
        - Serie de floats : [d1, d2, d3, ...]
          -> retourne directement la serie (cas fBm synthetique)

    La detection du format se fait sur le premier element :
        - float ou int -> serie directe
        - liste ou tuple -> points GPS

    Entree :
        points_ou_serie : liste de points GPS ou liste de float

    Sortie :
        list of float : distances en metres
    """
    if len(points_ou_serie) == 0:
        return []

    # Detection du format d'entree sur le premier element
    if isinstance(points_ou_serie[0], (int, float)):
        # Serie de floats -> fBm synthetique ou serie deja calculee
        return list(points_ou_serie)

    # Points GPS -> calcul des distances haversine
    distances = []
    taille    = len(points_ou_serie)

    for i in range(1, taille):
        d = haversine(points_ou_serie[i-1][0], points_ou_serie[i-1][1],
                      points_ou_serie[i][0],   points_ou_serie[i][1])
        distances.append(d * 1000)   # km -> metres

    return distances


def profil_cumule(distances):
    """
    Construit le profil cumule Y de la serie des distances

    Utilise par la methode DFA.

    Soit X = (x_i)_i=1..N et x_bar la moyenne de X.
    Le profil cumule est defini par :
        Y(k) = sum_{i=1}^{k} (x_i - x_bar), k = 1, ..., N

    Propriete : Y(N) = 0 par definition de la moyenne.

    Entree :
        distances : list of float (serie X)

    Sortie :
        Y : list of float de meme longueur que distances
    """
    taille = len(distances)

    # Calcul de la moyenne
    moy = 0
    for i in range(taille):
        moy += distances[i]
    moy = moy / taille

    # Construction du profil cumule
    Y   = []
    val = 0

    for i in range(taille):
        val += (distances[i] - moy)
        Y.append(val)

    return Y
