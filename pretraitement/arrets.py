# ./pretraitement/arrets.py
#
# DETECTION D'ARRETS, TEMPS DE SEJOUR ET DISTANCES ENTRE ARRETS
#
# Ce module fabrique les deux variables dont on etudie les lois de puissance
# (lois_de_puissance/) et dont depend le test du mecanisme ON/OFF (hypothese H2) :
#   - le TEMPS DE SEJOUR : duree pendant laquelle l'individu reste a peu pres au meme endroit ;
#   - la DISTANCE entre deux arrets consecutifs (et non les d_i entre points GPS successifs).
#
# ----------------------------------------------------------------------------------------
# Algorithme : fenetre glissante de Li et al. (2008), le standard de la litterature GPS
# ----------------------------------------------------------------------------------------
# Pour chaque point i pris comme ANCRE, on cherche le premier point j dont la distance a l'ancre
# depasse ARRET_RAYON_M. Si l'individu est reste dans le rayon pendant au moins ARRET_DUREE_MIN_S
# (t[j] - t[i]), on declare un arret : position = moyenne des points i..j-1, arrivee = t[i],
# depart = t[j] (premier instant ou on le voit ressortir). On reprend ensuite a j. Sinon l'ancre
# avance d'un point.
#
# Les seuils NE SONT PAS UNIVERSELS (voir documentation/LITTERATURE.md) : 100 m / 5 min (defaut de
# trackintel) ou 200 m / 20-30 min (Zheng et al., tres utilise sur GeoLife). Ils sont dans config.py
# (ARRET_*) et experiences/lois_de_puissance.py les fait varier pour mesurer ce qu'ils changent.
#
# ----------------------------------------------------------------------------------------
# Trois precautions contre les biais d'observation (Kivela & Porter, 2015)
# ----------------------------------------------------------------------------------------
# Une fenetre d'observation finie a plus de chances de commencer ou de finir PENDANT un long sejour
# que pendant un court : les longs sejours sont tronques, et la queue de la distribution est biaisee.
# On ne cache donc jamais ces cas, on les ETIQUETTE (colonne `censuree`) :
#   1. un arret qui commence au premier point d'un morceau continu est censure a gauche (l'appareil a
#      ete allume pendant le sejour : on ne sait pas depuis quand il dure) ;
#   2. un arret qui n'est jamais suivi d'une sortie du rayon n'est PAS detecte par defaut
#      (inclure_dernier=False), car sa fin est inconnue ; avec inclure_dernier=True il est garde et
#      etiquete censure a droite ;
#   3. un TROU d'enregistrement de plus de ARRET_TROU_MAX_S coupe la serie : on ne raccorde pas deux
#      morceaux separes par un trou (meme choix que trackintel, gap_threshold = 15 min).
# Les pauses entre deux segments (appareil eteint, puis rallume au meme endroit) sont recuperees a
# part, type "entre_segments", et sont TOUJOURS etiquetees censurees : leur duree est la duree du
# trou, pas celle du sejour.
#
# Entree standard : un DataFrame de points au format de sortie/<user>.csv (colonnes segment_id,
# timestamp, lat, lon ; les autres colonnes sont ignorees).

import numpy as np
import pandas as pd

from config import (ARRET_RAYON_M, ARRET_DUREE_MIN_S, ARRET_TROU_MAX_S,
                    ARRET_RAYON_PAUSE_M, ARRET_DISTANCE_MIN_M)
from pretraitement.geodesie import haversine_vectorise

# Colonnes du tableau des sejours, dans cet ordre.
COLONNES_SEJOURS = ["utilisateur", "segment_id", "type", "debut", "fin", "duree_s",
                    "lat", "lon", "n_points", "censuree"]

# Colonnes du tableau des distances entre arrets.
COLONNES_DISTANCES = ["utilisateur", "debut", "distance_m", "meme_segment"]


# ---------------------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------------------
def lire_points_sortie(chemin):
    """
    Lit un CSV de sortie/ (produit par 01_pretraitement.py) et ne garde que ce dont les arrets
    ont besoin.

    Entree :
        chemin : chemin de sortie/<user>.csv

    Sortie :
        DataFrame (segment_id, timestamp, lat, lon), trie par segment puis par temps.
    """
    points = pd.read_csv(chemin, usecols=["segment_id", "timestamp", "lat", "lon"])
    return points.sort_values(["segment_id", "timestamp"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Detection dans une serie continue
# ---------------------------------------------------------------------------------------
def _distances_a_l_ancre_m(lat, lon, i, debut, fin):
    """Distances en metres entre le point d'indice i et les points debut..fin-1."""
    return haversine_vectorise(lat[i], lon[i], lat[debut:fin], lon[debut:fin]) * 1000.0


def _premier_point_hors_rayon(lat, lon, i, rayon_m, taille_bloc=512):
    """
    Indice du premier point apres i qui est a plus de rayon_m du point i ; None si aucun.

    On avance par blocs de `taille_bloc` points au lieu de calculer toutes les distances d'un coup :
    un arret court est decide apres quelques points, inutile de mesurer jusqu'a la fin de la serie.
    """
    n = len(lat)
    debut = i + 1
    while debut < n:
        fin = min(debut + taille_bloc, n)
        hors = np.flatnonzero(_distances_a_l_ancre_m(lat, lon, i, debut, fin) > rayon_m)
        if hors.size > 0:
            return debut + int(hors[0])
        debut = fin
    return None


def _construire_arret(t, lat, lon, i, fin_exclue, t_fin, decalage, fin_observee):
    """
    Dictionnaire decrivant un arret fait des points i .. fin_exclue-1 d'un morceau.

    Entree :
        i, fin_exclue : bornes du morceau de points de l'arret (fin exclue)
        t_fin         : instant de depart (t du premier point hors du rayon, ou dernier t si fin inconnue)
        decalage      : position du morceau dans la serie complete (pour retrouver les indices globaux)
        fin_observee  : False si on n'a jamais vu l'individu ressortir du rayon (censure a droite)
    """
    return dict(
        debut=float(t[i]),
        fin=float(t_fin),
        duree_s=float(t_fin - t[i]),
        lat=float(np.mean(lat[i:fin_exclue])),       # centroide : moyenne arithmetique, suffisante a l'echelle du rayon
        lon=float(np.mean(lon[i:fin_exclue])),
        n_points=int(fin_exclue - i),
        indice_debut=int(decalage + i),
        indice_fin=int(decalage + fin_exclue - 1),
        censuree_debut=bool(i == 0),                 # commence au premier point : debut du sejour inconnu
        censuree_fin=bool(not fin_observee),
    )


def _arrets_dans_morceau(t, lat, lon, decalage, rayon_m, duree_min_s, inclure_dernier):
    """Applique la fenetre glissante de Li et al. a UN morceau continu (sans trou)."""
    n = len(t)
    arrets = []
    i = 0
    while i < n - 1:
        j = _premier_point_hors_rayon(lat, lon, i, rayon_m)
        if j is None:
            # Aucun point ne sort du rayon : le candidat va jusqu'a la fin du morceau, sa fin est
            # inconnue. On ne le garde que si on l'a demande, et on l'etiquette censure a droite.
            if inclure_dernier and (t[n - 1] - t[i]) >= duree_min_s:
                arrets.append(_construire_arret(t, lat, lon, i, n, t[n - 1], decalage, fin_observee=False))
            break
        if (t[j] - t[i]) >= duree_min_s:
            arrets.append(_construire_arret(t, lat, lon, i, j, t[j], decalage, fin_observee=True))
            i = j                                    # on reprend apres la sortie
        else:
            i += 1                                   # trop court : l'ancre avance d'un point
    return arrets


def detecter_arrets(t, lat, lon, rayon_m=ARRET_RAYON_M, duree_min_s=ARRET_DUREE_MIN_S,
                    trou_max_s=ARRET_TROU_MAX_S, inclure_dernier=False):
    """
    Detecte les arrets dans une serie de points GPS (algorithme de Li et al., voir en-tete).

    Entree :
        t, lat, lon     : tableaux de meme longueur ; t en secondes, CROISSANT
        rayon_m         : rayon de l'arret (m)
        duree_min_s     : duree minimale d'un arret (s)
        trou_max_s      : un intervalle plus long que ceci coupe la serie en morceaux independants
        inclure_dernier : garder l'eventuel arret final jamais termine (censure a droite)

    Sortie :
        liste de dictionnaires, dans l'ordre du temps. Cles : debut, fin, duree_s, lat, lon,
        n_points, indice_debut, indice_fin, censuree_debut, censuree_fin.
    """
    t = np.asarray(t, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if not (len(t) == len(lat) == len(lon)):
        raise ValueError("t, lat et lon doivent avoir la meme longueur")
    n = len(t)
    if n < 2:
        return []
    # Morceaux continus : on coupe a chaque trou d'enregistrement, sans raccorder.
    coupures = np.flatnonzero(np.diff(t) > trou_max_s) + 1
    bornes = np.concatenate([[0], coupures, [n]])
    arrets = []
    for a, b in zip(bornes[:-1], bornes[1:]):
        arrets += _arrets_dans_morceau(t[a:b], lat[a:b], lon[a:b], int(a), rayon_m, duree_min_s, inclure_dernier)
    return arrets


# ---------------------------------------------------------------------------------------
# Sejours d'un utilisateur
# ---------------------------------------------------------------------------------------
def sejours_utilisateur(points, utilisateur="", rayon_m=ARRET_RAYON_M, duree_min_s=ARRET_DUREE_MIN_S,
                        trou_max_s=ARRET_TROU_MAX_S, rayon_pause_m=ARRET_RAYON_PAUSE_M,
                        inclure_dernier=False):
    """
    Tous les sejours d'un utilisateur : arrets INTERNES aux segments et pauses ENTRE segments.

    Entree :
        points        : DataFrame (segment_id, timestamp, lat, lon) -- voir lire_points_sortie()
        utilisateur   : identifiant a inscrire dans la colonne `utilisateur`
        rayon_m, duree_min_s, trou_max_s, inclure_dernier : voir detecter_arrets()
        rayon_pause_m : une pause entre deux segments consecutifs n'est retenue que si la fin du
                        premier et le debut du second sont a moins de ce rayon (m)

    Sortie :
        DataFrame trie par `debut`, colonnes COLONNES_SEJOURS :
          type      "interne" (arret vu a l'interieur d'un segment) ou "entre_segments"
          censuree  True si la duree est incertaine : debut ou fin inconnu pour un arret interne ;
                    TOUJOURS True pour une pause entre segments (duree = duree du trou)
        Limite a connaitre : entre deux segments RETENUS se trouvent parfois des segments ecartes par
        le pretraitement (trop courts) ; une pause "entre_segments" peut alors cacher un deplacement.
    """
    requises = {"segment_id", "timestamp", "lat", "lon"}
    if not requises <= set(points.columns):
        raise ValueError(f"colonnes manquantes : {sorted(requises - set(points.columns))}")

    lignes = []
    bornes_segments = []   # (segment_id, t_debut, t_fin, lat_debut, lon_debut, lat_fin, lon_fin)
    for seg_id, g in points.groupby("segment_id", sort=True):
        g = g.sort_values("timestamp", kind="stable")
        t = g["timestamp"].to_numpy(dtype=float)
        lat = g["lat"].to_numpy(dtype=float)
        lon = g["lon"].to_numpy(dtype=float)
        for a in detecter_arrets(t, lat, lon, rayon_m, duree_min_s, trou_max_s, inclure_dernier):
            lignes.append(dict(
                utilisateur=utilisateur, segment_id=int(seg_id), type="interne",
                debut=a["debut"], fin=a["fin"], duree_s=a["duree_s"], lat=a["lat"], lon=a["lon"],
                n_points=a["n_points"], censuree=bool(a["censuree_debut"] or a["censuree_fin"])))
        if len(t) > 0:
            bornes_segments.append((int(seg_id), t[0], t[-1], lat[0], lon[0], lat[-1], lon[-1]))

    # Pauses entre segments consecutifs (dans l'ordre du temps) : appareil eteint puis rallume au
    # meme endroit. Duree = duree du trou, donc toujours censuree.
    bornes_segments.sort(key=lambda b: b[1])
    for prec, suiv in zip(bornes_segments[:-1], bornes_segments[1:]):
        trou = suiv[1] - prec[2]
        if trou < duree_min_s:
            continue
        ecart_m = float(haversine_vectorise(prec[5], prec[6], suiv[3], suiv[4]) * 1000.0)
        if ecart_m <= rayon_pause_m:
            lignes.append(dict(
                utilisateur=utilisateur, segment_id=prec[0], type="entre_segments",
                debut=float(prec[2]), fin=float(suiv[1]), duree_s=float(trou),
                lat=float((prec[5] + suiv[3]) / 2), lon=float((prec[6] + suiv[4]) / 2),
                n_points=0, censuree=True))

    sejours = pd.DataFrame(lignes, columns=COLONNES_SEJOURS)
    return sejours.sort_values("debut", kind="stable").reset_index(drop=True)


def temps_de_sejour(sejours, seulement_non_censurees=True):
    """
    Durees de sejour (s), pretes pour l'ajustement d'une loi de puissance.

    Par defaut : arrets internes NON censures seulement. C'est le choix prudent : un sejour dont le
    debut ou la fin est inconnu est plus court que la realite (biais de la fenetre d'observation),
    et les pauses entre segments mesurent la duree d'un trou. Avec seulement_non_censurees=False on
    garde tous les arrets internes ET les pauses entre segments : sert a la comparaison de sensibilite.

    Sortie : tableau numpy de floats (vide si aucun sejour).
    """
    if len(sejours) == 0:
        return np.array([], dtype=float)
    if seulement_non_censurees:
        retenus = sejours[(sejours["type"] == "interne") & (~sejours["censuree"].astype(bool))]
    else:
        retenus = sejours
    return retenus["duree_s"].to_numpy(dtype=float)


def distances_entre_arrets(sejours, meme_segment_seulement=True, distance_min_m=ARRET_DISTANCE_MIN_M):
    """
    Distances (m) entre arrets CONSECUTIFS dans le temps : la variable "distance" des lois de puissance.

    Ce ne sont pas les d_i entre points GPS successifs : on mesure le deplacement d'un lieu de sejour
    au suivant (comme Wang et al. et Alessandretti et al.).

    Entree :
        sejours                : DataFrame de sejours_utilisateur() (un seul utilisateur, ou plusieurs
                                 dont les paires ne sont comparees qu'au sein d'un meme utilisateur)
        meme_segment_seulement : ne garder que les paires dont les deux arrets sont internes au MEME
                                 segment (aucun trou entre eux : le deplacement est vraiment observe).
                                 Si False, on garde aussi les paires a cheval sur un trou : plus de
                                 donnees, mais des arrets intermediaires ont pu passer inapercus,
                                 ce qui gonfle les distances.
        distance_min_m         : on ecarte les paires plus proches que ce seuil (deux arrets au meme
                                 endroit ne sont pas un deplacement)

    Sortie : DataFrame COLONNES_DISTANCES (utilisateur, debut = depart du premier arret,
             distance_m, meme_segment).
    """
    vide = pd.DataFrame(columns=COLONNES_DISTANCES)
    if len(sejours) < 2:
        return vide
    parts = []
    for utilisateur, s in sejours.groupby("utilisateur", sort=True):
        s = s.sort_values("debut", kind="stable")
        if len(s) < 2:
            continue
        lat = s["lat"].to_numpy(dtype=float)
        lon = s["lon"].to_numpy(dtype=float)
        distances = haversine_vectorise(lat[:-1], lon[:-1], lat[1:], lon[1:]) * 1000.0
        types = s["type"].to_numpy()
        segments = s["segment_id"].to_numpy()
        meme = (types[:-1] == "interne") & (types[1:] == "interne") & (segments[:-1] == segments[1:])
        parts.append(pd.DataFrame({
            "utilisateur": utilisateur,
            "debut": s["fin"].to_numpy(dtype=float)[:-1],
            "distance_m": distances,
            "meme_segment": meme}))
    if not parts:
        return vide
    sortie = pd.concat(parts, ignore_index=True)
    sortie = sortie[sortie["distance_m"] > distance_min_m]
    if meme_segment_seulement:
        sortie = sortie[sortie["meme_segment"]]
    return sortie[COLONNES_DISTANCES].reset_index(drop=True)
