# ./pretraitement/segmentation.py

import numpy as np

from pretraitement.geodesie import haversine_vectorise


def segmenter_points(points, seuil_temps):
    """
    Découpe une liste de points en segments selon un seuil temporel

    Si l'écart entre deux points successifs dépasse seuil_temps,
    on coupe et on commence un nouveau segment.

    Entrée :
        points      : [(lat, lon, alt, timestamp), ...]
        seuil_temps : float, en secondes (ex: 600 = 10 minutes)

    Sortie :
        list of list : [ [points_seg1], [points_seg2], ... ]
    """
    if len(points) == 0:
        return []

    segments       = []
    segment_courant = [points[0]]

    for i in range(1, len(points)):
        t1 = points[i - 1][3]
        t2 = points[i][3]

        delta_t = t2 - t1

        if delta_t > seuil_temps:
            segments.append(segment_courant)
            segment_courant = [points[i]]
        else:
            segment_courant.append(points[i])

    # Ne pas oublier le dernier segment en cours
    segments.append(segment_courant)

    return segments


def filtrer_segments(segments, min_points):
    """
    Supprime les segments trop courts (pas assez de points)

    Un segment trop court n'est pas exploitable pour estimer H.

    Fonction inchangée par rapport à l'ancien pipeline : len(segment)
    fonctionne aussi bien sur une liste de tuples que sur un DataFrame
    pandas (nb de lignes), donc aucune adaptation n'était nécessaire
    pour s'appliquer désormais en étape 7, après le marquage de
    Phase 3 (cf. main.py).

    Entrée :
        segments   : liste de segments (listes de tuples ou DataFrames)
        min_points : int, nombre minimum de points requis (ex: 50)

    Sortie :
        list : segments conservés
    """
    segments_valides = []

    for segment in segments:
        if len(segment) >= min_points:
            segments_valides.append(segment)

    return segments_valides


def filtrer_segments_stationnaires(segments, distance_min_km=None):
    """
    Supprime les segments où la personne n'a pas vraiment bougé.

    La distance totale parcourue est calculée comme la somme des
    distances entre points successifs. Si elle est inférieure au
    seuil effectif, le segment est considéré stationnaire.

    ─────────────────────────────────────────────────────────────────
    V5 : seuil PAR SEGMENT (Option B du plan V5)
    ─────────────────────────────────────────────────────────────────
    Le seuil effectif n'est plus UN SEUL float global : chaque
    segment peut porter son propre seuil dans
        segment.attrs["distance_min_km"]
    (attribut posé par 01_pretraitement.py, qui a accès au pas
    médian du segment via ses stats Phase 2).

    Priorité de lecture du seuil, par segment :
        1. segment.attrs["distance_min_km"] s'il existe (V5, Option B)
        2. sinon, distance_min_km global passé en argument (V4, fallback)

    Ce mécanisme à deux niveaux garantit :
        - la PARITÉ V4 : les appelants qui ne posent pas l'attribut
          (tous ceux qui ne sont pas 01_pretraitement.py) obtiennent
          exactement le comportement V4 ;
        - la FLEXIBILITÉ V5 : un segment de marche (pas médian 5 s)
          et un segment de bus (pas médian 3 s) peuvent recevoir
          des seuils différents, calculés en amont à partir de leur
          propre pas médian.

    Si AUCUN seuil n'est disponible (ni attrs ni argument), le
    segment est conservé par défaut (comportement conservateur :
    mieux vaut un segment de plus qu'un segment perdu par erreur).

    Décision des documents de réflexion : DISTANCE_MIN reste inchangé
    en valeur par défaut (0.1 km) ; seule la façon de le calculer
    change (01_pretraitement.py peut le dériver du pas médian).
    Cette fonction ne fait qu'appliquer le seuil qu'on lui donne.

    Entrée :
        segments        : liste de DataFrames (un par segment, colonnes
                          au moins "lat", "lon"), optionnellement avec
                          segment.attrs["distance_min_km"] (float, km)
        distance_min_km : float ou None, seuil global de secours (km).
                          V4 : obligatoire. V5 : optionnel (les seuils
                          par segment sont dans attrs).

    Sortie :
        list : segments (DataFrames) non stationnaires
    """
    segments_valides = []

    for segment in segments:
        # Lecture du seuil PAR SEGMENT (V5), avec fallback global (V4).
        seuil = None
        if hasattr(segment, "attrs"):
            seuil = segment.attrs.get("distance_min_km")
        if seuil is None:
            seuil = distance_min_km

        # Aucun seuil disponible : on conserve le segment par défaut
        # (prudence : ne jamais éliminer un segment "par accident").
        if seuil is None:
            segments_valides.append(segment)
            continue

        lat = segment["lat"].to_numpy()
        lon = segment["lon"].to_numpy()

        if len(lat) < 2:
            distance_totale = 0.0
        else:
            # haversine_vectorise() calcule toutes les distances entre
            # points consécutifs en une seule opération (lat[:-1] vs
            # lat[1:]), puis np.sum() les additionne : remplace la
            # boucle "for i in range(1, len(segment)): distance_totale
            # += haversine(...)" de l'ancienne implémentation.
            distance_totale = float(np.sum(
                haversine_vectorise(lat[:-1], lon[:-1], lat[1:], lon[1:])
            ))

        if distance_totale >= seuil:
            segments_valides.append(segment)

    return segments_valides


# ============================================================================
# AJOUT V5 -- coupure aux points marques (OPTION)
# ----------------------------------------------------------------------------
# La V4 MARQUE les points suspects (MAD) sans jamais les retirer : un pic GPS isole reste
# dans la serie des d_i et peut dominer l'estimation de H (ex. un saut de 1,8 km dans un segment
# de marche a 1,5 m/s). Cette fonction propose l'alternative : COUPER le segment aux points
# marques, SANS RACCORDER les deux morceaux (raccorder fabriquerait un saut artificiel qui
# deforme la DFA : Chen et al., 2002 -- c'est le defaut de la variante "H sans suspects" de la V4).
#
# Les points "regime" (changement de mode : marche -> bus) ne sont JAMAIS des points de coupure :
# ils sont du comportement reel, pas des artefacts.
#
# Cette fonction ne change rien au pretraitement (01_pretraitement.py) ni aux CSV de sortie ;
# elle est appelee par l'estimation (colonne h_dfa_morceaux) et par
# experiences/sensibilite_pretraitement.py qui mesure son effet sur H.
# ============================================================================

def couper_aux_points_marques(df, min_points):
    """
    Decoupe un segment marque en morceaux continus, en retirant les points 'artefact' ou 'suspect_dist'.

    Entree :
        df         : DataFrame du segment avec les colonnes de marquage de la Phase 3
                     ("suspect_dist", "type_suspect_vitesse"), lat, lon, timestamp...
        min_points : taille minimale d'un morceau conserve

    Sortie :
        list de DataFrames : morceaux continus de au moins min_points points. Si aucun point
        n'est a couper, renvoie [df] (meme objet) s'il est assez long.
    """
    mauvais = np.zeros(len(df), dtype=bool)
    if "suspect_dist" in df.columns:
        mauvais |= df["suspect_dist"].fillna(False).to_numpy(dtype=bool)
    if "type_suspect_vitesse" in df.columns:
        mauvais |= (df["type_suspect_vitesse"].fillna("").to_numpy() == "artefact")
    if not mauvais.any():
        return [df] if len(df) >= min_points else []
    morceaux, debut = [], 0
    for i in np.flatnonzero(mauvais):
        if i > debut:
            morceaux.append(df.iloc[debut:i])
        debut = i + 1
    if debut < len(df):
        morceaux.append(df.iloc[debut:])
    return [m for m in morceaux if len(m) >= min_points]