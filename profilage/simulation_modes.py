# ./profilage/simulation_modes.py
#
# Segments SYNTHETIQUES de 5 "modes" aux structures de memoire differentes, pour valider la CHAINE de profilage
# (caracteristiques -> clustering -> classification) sur une verite connue.
#
# AVERTISSEMENT DE PORTEE (a dire a l'oral) : ces classes sont CONSTRUITES pour differer par la memoire ; un bon
# score ici montre que la chaine sait retrouver une structure de memoire quand elle existe, PAS que les modes
# de transport reels de GeoLife different par leur memoire. Le niveau moyen de d_i differe aussi entre classes mais
# n'est PAS dans le vecteur de caracteristiques (H, alpha, S, D_f sont invariants d'echelle).
#
#   marche   : d_i = niveau * (1 + 0.25 AR(1) phi = 0.4)                      memoire courte
#   velo     : d_i = niveau * (1 + 0.25 fGn H = 0.65)                          memoire longue moderee
#   bus      : ON/OFF a durees de Pareto (alpha = 1.5, xm = 3) ; ON = niveau    memoire longue par alternance arret/mouvement
#   voiture  : changements de regime geometriques (0, 0.4, 1 x niveau)        persistance SPURIEUSE (regimes)
#   train    : d_i = niveau * (1 + 0.10 fGn H = 0.85)                          memoire longue forte, tres regulier
# Le cap suit une marche aleatoire de variance kappa (plus petit = trajectoire plus droite).

import numpy as np

from simulation import bruit_gaussien_fractionnaire, ar1, on_off, changements_de_regime

CLASSES = ["marche", "velo", "bus", "voiture", "train"]
NIVEAU_M = dict(marche=6.5, velo=20.0, bus=30.0, voiture=60.0, train=80.0)          # metres par pas de 5 s
KAPPA = dict(marche=0.35, velo=0.20, bus=0.12, voiture=0.10, train=0.02)
PAS_S = 5.0


def serie_distances(classe, n, rng, echelle=1.0):
    niv = NIVEAU_M[classe] * echelle
    if classe == "marche":
        d = niv * (1.0 + 0.25 * ar1(n, 0.4, rng))
    elif classe == "velo":
        d = niv * (1.0 + 0.25 * bruit_gaussien_fractionnaire(n, 0.65, rng))
    elif classe == "bus":
        etat = on_off(n, 1.5, 1.5, rng, xm=3.0)
        d = etat * niv * (1.0 + 0.2 * ar1(n, 0.5, rng)) + (1 - etat) * 0.3 * np.abs(rng.standard_normal(n))
    elif classe == "voiture":
        regime = changements_de_regime(n, rng, niveaux=(0.0, 0.4, 1.0), duree_moyenne=40.0, bruit=0.0)
        d = niv * regime * (1.0 + 0.1 * rng.standard_normal(n))
    elif classe == "train":
        d = niv * (1.0 + 0.10 * bruit_gaussien_fractionnaire(n, 0.85, rng))
    else:
        raise ValueError(classe)
    return np.maximum(d, 0.0)


def generer_segment(classe, n, rng, echelle=1.0, lat0=39.9, lon0=116.4):
    """Sortie : dict(d (n,), v (n,) km/h, lat (n+1,), lon (n+1,), classe)."""
    d = serie_distances(classe, n, rng, echelle)
    cap = np.cumsum(KAPPA[classe] * rng.standard_normal(n)) + rng.uniform(0, 2 * np.pi)
    dx = np.r_[0.0, np.cumsum(d * np.cos(cap))]
    dy = np.r_[0.0, np.cumsum(d * np.sin(cap))]
    lat = lat0 + dy / 110574.0
    lon = lon0 + dx / (111320.0 * np.cos(np.deg2rad(lat0)))
    return dict(d=d, v=d / PAS_S * 3.6, lat=lat, lon=lon, classe=classe)


def generer_jeu(n_utilisateurs=24, segments_par_classe=5, graine=20261018, longueur_mediane=700, longueur_min=300,
                longueur_max=2500):
    """
    Liste de segments, chacun avec `utilisateur` et `classe` (int). Chaque utilisateur a un facteur d'echelle
    lognormal (heterogeneite inter-utilisateurs : la validation croisee PAR UTILISATEUR est donc necessaire).
    """
    rng = np.random.default_rng(graine)
    jeu = []
    for u in range(n_utilisateurs):
        echelle = float(np.exp(0.15 * rng.standard_normal()))
        for c, nom in enumerate(CLASSES):
            for _ in range(segments_par_classe):
                n = int(np.clip(np.exp(np.log(longueur_mediane) + 0.5 * rng.standard_normal()), longueur_min, longueur_max))
                s = generer_segment(nom, n, rng, echelle)
                s.update(utilisateur=u, y=c)
                jeu.append(s)
    return jeu
