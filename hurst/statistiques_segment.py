# ./hurst/statistiques_segment.py
#
# Statistiques descriptives d'un segment GPS et leur affichage, utilisees par les
# scripts d'estimation. (Ex-moitie de stats_segment_gps.py)

from pretraitement.geodesie import haversine


def statistiques_segment(segment):
    """
    Calcule les statistiques descriptives d'un segment GPS

    Ces statistiques contextualisent l'estimation de H et
    permettent de detecter les segments contamines par des
    sauts GPS (dist_max anormalement elevee).

    Entree :
        segment : liste de points [[lat, lon, timestamp], ...]
                  le timestamp est a l'indice 2 (ou 3 si alt presente)

    Sortie :
        dict : {
            'nb_points'       : int,
            'distance_totale' : float (km),
            'distance_nette'  : float (km),
            'dist_max'        : float (metres),
            'variance_dist'   : float,
            'delta_t_moyen_s' : float ou None,
            'vitesse_max_kmh' : float ou None
        }
    """
    taille    = len(segment)
    distances = []
    delta_ts  = []

    for i in range(1, taille):
        d = haversine(segment[i-1][0], segment[i-1][1],
                      segment[i][0],   segment[i][1])
        distances.append(d)

        # Cherche le timestamp : indice 2 si [lat,lon,t]
        #                        indice 3 si [lat,lon,alt,t]
        idx_t = 3 if len(segment[i]) >= 4 else 2

        if len(segment[i]) > 2:
            dt = segment[i][idx_t] - segment[i-1][idx_t]
            delta_ts.append(dt)

    N = len(distances)

    if N == 0:
        return {
            'nb_points'       : taille,
            'distance_totale' : 0.0,
            'distance_nette'  : 0.0,
            'dist_max'        : 0.0,
            'variance_dist'   : 0.0,
            'delta_t_moyen_s' : None,
            'vitesse_max_kmh' : None
        }

    # Distance totale et nette
    distance_totale = 0.0
    for d in distances:
        distance_totale += d

    distance_nette = haversine(segment[0][0],  segment[0][1],
                               segment[-1][0], segment[-1][1])

    # Distance maximale inter-points (signal de saut GPS si elevee)
    dist_max = 0.0
    for d in distances:
        if d > dist_max:
            dist_max = d

    # Variance des distances
    moyenne_dist = distance_totale / N
    variance     = 0.0
    for d in distances:
        ecart     = d - moyenne_dist
        variance += ecart * ecart
    variance = variance / N

    # Delta-t moyen et vitesse max (necessitent le timestamp)
    delta_t_moyen   = None
    vitesse_max_kmh = None

    if len(delta_ts) > 0:
        somme_dt = 0.0
        for dt in delta_ts:
            somme_dt += dt
        delta_t_moyen = somme_dt / len(delta_ts)

        vitesses = []
        for i in range(len(delta_ts)):
            if delta_ts[i] > 0:
                vitesses.append(distances[i] / (delta_ts[i] / 3600))

        if len(vitesses) > 0:
            vitesse_max_kmh = 0.0
            for v in vitesses:
                if v > vitesse_max_kmh:
                    vitesse_max_kmh = v

    return {
        'nb_points'       : taille,
        'distance_totale' : round(distance_totale, 4),
        'distance_nette'  : round(distance_nette, 4),
        'dist_max'        : round(dist_max * 1000, 2),   # metres
        'variance_dist'   : round(variance, 6),
        'delta_t_moyen_s' : round(delta_t_moyen, 2)   if delta_t_moyen   is not None else None,
        'vitesse_max_kmh' : round(vitesse_max_kmh, 2) if vitesse_max_kmh is not None else None
    }


def afficher_statistiques(segment_id, H, stats):
    """
    Affiche les statistiques d'un segment de maniere lisible

    Entree :
        segment_id : int, identifiant du segment
        H          : float, estimation de H (ou None)
        stats      : dict, retour de statistiques_segment()
    """
    h_str = f"{H:.4f}" if H is not None else "N/A"

    print(f"┌─ Segment {segment_id} {'─'*23}")
    print(f"│  H estime         : {h_str}")
    print(f"│  Nb points        : {stats['nb_points']}")
    print(f"│  Distance totale  : {stats['distance_totale']*1000:.1f} m")
    print(f"│  Distance nette   : {stats['distance_nette']*1000:.1f} m")
    print(f"│  Distance max     : {stats['dist_max']:.2f} m")
    print(f"│  Variance dist    : {stats['variance_dist']:.6f}")

    if stats['delta_t_moyen_s'] is not None:
        print(f"│  Delta-t moyen    : {stats['delta_t_moyen_s']:.1f} s")

    if stats['vitesse_max_kmh'] is not None:
        print(f"│  Vitesse max      : {stats['vitesse_max_kmh']:.1f} km/h")

    print(f"└{'─'*36}")
