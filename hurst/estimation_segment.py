# ./hurst/estimation_segment.py
#
# Assemble les trois methodes (DFA, R/S, Whittle local), le test nul a memoire courte et les
# diagnostics d'echantillonnage pour UN segment : une ligne de resultats par segment.
#
# Entree standard : points = [[lat, lon, timestamp], ...] (comme DFA(), RS(), Whittle() de la V4).
# Chaque resultat porte l'empreinte de config.py : on sait avec quels reglages il a ete produit.

import zlib

import numpy as np

from config import (GRAINE, SERIE_ANALYSEE, AVEC_TEST_NUL, NUL_ORDRE_AR_MAX, N_MIN_WHITTLE, empreinte_config)
from hurst.dfa import dfa_complet, dfa_h
from hurst.rs import rs_complet
from hurst.whittle import whittle_local
from pretraitement.echantillonnage import profil_echantillonnage
from pretraitement.geodesie import haversine_vectorise
from pretraitement.segmentation import couper_aux_points_marques
from validation.modeles_nuls import evaluer_depassement_dfa


def graine_segment(utilisateur, segment_id):
    """
    Graine propre a un segment : le test nul d'un segment ne depend que de (GRAINE, utilisateur,
    segment), jamais de l'ordre de traitement ni de la reprise apres interruption.
    """
    return np.random.SeedSequence([GRAINE, zlib.crc32(str(utilisateur).encode()), int(segment_id)])


def series_du_segment(points):
    """
    Entree : points = [[lat, lon, timestamp], ...]
    Sortie : dict de tableaux numpy
        t   timestamps            dt  ecarts de temps (s)
        d   distances d_i (m)     v   vitesses v_i = d_i / dt_i (km/h)
    """
    a = np.asarray(points, dtype=float)
    t = a[:, 2]
    dt = np.diff(t)
    d = haversine_vectorise(a[:-1, 0], a[:-1, 1], a[1:, 0], a[1:, 1]) * 1000.0     # km -> m
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dt > 0, d / dt * 3.6, np.nan)
    return dict(t=t, dt=dt, d=d, v=v)


def estimer_serie(x, rng, avec_nul=AVEC_TEST_NUL, ordre_ar_max=NUL_ORDRE_AR_MAX):
    """
    Estimateurs et test nul sur une serie numerique quelconque (sans notion de GPS).
    ordre_ar_max : ordre maximal du modele nul AR(p). Presenter p = 1 ET p = 5 : p = 5 est conservateur
    (peu de faux positifs, peu de puissance), p = 1 est puissant mais legerement liberal.
    """
    dfa = dfa_complet(x)
    rs = rs_complet(x)
    w = whittle_local(x)
    sortie = dict(
        n_serie=len(x), h_dfa=dfa.h, dfa_bas=dfa.h_bas, dfa_haut=dfa.h_haut, ecart_pentes=dfa.ecart_pentes,
        dfa_sup_1=bool(dfa.h > 1) if np.isfinite(dfa.h) else False,
        h_rs_brut=rs.h_brut, h_rs_corrige=rs.h_corrige,
        h_whittle=w.h, whittle_m=w.m, whittle_a_la_borne=w.a_la_borne,
    )
    if avec_nul:
        t = evaluer_depassement_dfa(x, rng, ordre_ar_max=ordre_ar_max)
        sortie.update(nul_p=t["p"], nul_seuil=t["nul_seuil"], nul_exces=t["exces"], nul_ordre_ar=ordre_ar_max)
    return sortie


def estimer_segment(points, utilisateur="", segment_id=0, serie=SERIE_ANALYSEE, avec_nul=AVEC_TEST_NUL,
                    ordre_ar_max=NUL_ORDRE_AR_MAX):
    """
    Une ligne de resultats pour un segment.

    serie : "distance" (d_i, m) ou "vitesse" (v_i, km/h). Les diagnostics d'echantillonnage
    (mode, cv_dt, corr_d_dt) servent a tester si H depend du logger plutot que du deplacement (H3).
    """
    s = series_du_segment(points)
    d, dt = s["d"], s["dt"]
    if serie == "distance":
        x = d
    elif serie == "vitesse":
        x = s["v"][np.isfinite(s["v"])]
    else:
        raise ValueError(f"serie inconnue : {serie!r} (attendu 'distance' ou 'vitesse')")
    profil = profil_echantillonnage(dt, d)
    trajet = float(d.sum())
    a = np.asarray(points, dtype=float)
    net = float(haversine_vectorise(a[0, 0], a[0, 1], a[-1, 0], a[-1, 1]) * 1000.0)
    ligne = dict(
        n_points=len(points), serie=serie, duree_s=float(s["t"][-1] - s["t"][0]), trajet_m=trajet,
        rapport_net=net / trajet if trajet > 0 else float("nan"),
        cv_d=float(d.std() / d.mean()) if d.mean() > 0 else float("nan"),
        part_immobile=float(np.mean(d < 1.0)),
        mode_echantillonnage=profil["mode"], pas_median_s=profil["pas_median"],
        cv_dt=profil["cv_dt"], corr_d_dt=profil["corr_d_dt"],
    )
    rng = np.random.default_rng(graine_segment(utilisateur, segment_id))
    ligne.update(estimer_serie(x, rng, avec_nul, ordre_ar_max))
    ligne["empreinte_config"] = empreinte_config()
    return ligne


def h_dfa_par_morceaux(df_marque, min_points=N_MIN_WHITTLE):
    """
    H (DFA) apres COUPURE aux points 'artefact' / 'suspect_dist', sans raccorder les morceaux.

    Remplace "H sans suspects" de la V4, qui retirait des points puis recalculait les distances
    entre voisins restants (saut artificiel). Ici chaque morceau est analyse separement et
    on combine par moyenne ponderee par la longueur.

    Entree : df_marque, DataFrame du segment (lat, lon, timestamp, suspect_dist, type_suspect_vitesse)
    Sortie : (h_moyen_pondere, nb_morceaux, nb_points_conserves) ; h = None si aucun morceau exploitable.
    """
    morceaux = couper_aux_points_marques(df_marque, min_points)
    valeurs, poids = [], []
    for m in morceaux:
        pts = m[["lat", "lon", "timestamp"]].to_numpy(dtype=float)
        h = dfa_h(series_du_segment(pts)["d"])
        if np.isfinite(h):
            valeurs.append(h)
            poids.append(len(m))
    if not valeurs:
        return None, len(morceaux), int(sum(len(m) for m in morceaux))
    return float(np.average(valeurs, weights=poids)), len(morceaux), int(sum(poids))
