"""Outils communs aux tests : fabriques de fichiers .plt et de segments marques synthetiques."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from pretraitement.filtres import filtrer_saut_distance, filtrer_vitesse
from pretraitement.stats_segment import calculer_stats_segment, segment_en_dataframe

ENTETE_PLT = ["Geolife trajectory", "WGS 84", "Altitude is in Feet", "Reserved 3",
              "0,2,255,My Track,0,0,2,8421376", "0"]
T0 = 1_224_730_384.0   # 2008-10-23 02:53:04 UTC


def ecrire_plt(chemin, t, lat, lon, lignes_en_plus=()):
    """Ecrit un .plt GeoLife valide : t en secondes UTC, lat/lon en degres."""
    lignes = list(ENTETE_PLT)
    for ti, la, lo in zip(t, lat, lon):
        d = datetime.fromtimestamp(float(ti), tz=timezone.utc)
        lignes.append(f"{la:.6f},{lo:.6f},0,100,40000.0,{d:%Y-%m-%d},{d:%H:%M:%S}")
    lignes += list(lignes_en_plus)
    Path(chemin).write_text("\n".join(lignes) + "\n", encoding="utf-8")


def marche(n, pas_m=1.4, dt=1.0, t0=T0, lat0=39.99, lon0=116.30):
    """Marche rectiligne vers le nord a vitesse constante : (t, lat, lon)."""
    t = t0 + dt * np.arange(n)
    return t, lat0 + (pas_m / 111_194.9) * np.arange(n), np.full(n, lon0)


def segment_marque(vitesses_ms, dt, t0=1.2e9, lon=116.3):
    """
    Segment rectiligne de vitesses donnees, passe par le marquage MAD de la Phase 3.
    Renvoie le DataFrame marque (suspect_dist, suspect_vitesse, type_suspect_vitesse, alerte_heterogene).
    """
    t = t0 + np.cumsum(dt)
    lat = 39.9 + np.cumsum(vitesses_ms * dt) / 111_194.9
    seg = [(float(a), float(lon), 0.0, float(c)) for a, c in zip(lat, t)]
    stats = calculer_stats_segment(seg)
    df = segment_en_dataframe(seg)
    df = filtrer_saut_distance(df, config.MAX_DIST_M, mode=config.MODE_MAX_DIST)
    return filtrer_vitesse(df, config.MAX_SPEED, stats_segment=stats)


def points_gps(t, lat, lon):
    """Format attendu par DFA(), RS(), Whittle() et estimer_segment() : [[lat, lon, timestamp], ...]."""
    return [[float(a), float(b), float(c)] for a, b, c in zip(lat, lon, t)]


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def ecrire_sortie(dossier, utilisateurs=("000", "001"), nb_segments=4, n=320, graine=11):
    """
    Ecrit des CSV de sortie au format de la V4 (segment_id, point_id, lat, lon, alt, timestamp,
    suspect_dist, suspect_vitesse, type_suspect_vitesse, alerte_heterogene), avec des vitesses
    persistantes et un artefact par segment sur deux.
    """
    from simulation.bruit_fractionnaire import bruit_gaussien_fractionnaire
    from simulation.marginale_lognormale import rendre_positive
    rng = np.random.default_rng(graine)
    Path(dossier).mkdir(parents=True, exist_ok=True)
    for u in utilisateurs:
        blocs = []
        for sid in range(nb_segments):
            v = rendre_positive(bruit_gaussien_fractionnaire(n, 0.7, rng), 0.5) * 1.4
            if sid % 2 == 0:
                v[n // 2] = 60.0
            df = segment_marque(v, rng.uniform(1, 4, n), t0=1.2e9 + sid * 1e5)
            df.insert(0, "point_id", np.arange(n))
            df.insert(0, "segment_id", sid)
            blocs.append(df)
        pd.concat(blocs, ignore_index=True).to_csv(Path(dossier) / f"{u}.csv", index=False)
    return Path(dossier)
