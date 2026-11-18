import numpy as np
import pandas as pd

from conftest import marche, segment_marque
from pretraitement.segmentation import (couper_aux_points_marques, filtrer_segments,
                                        filtrer_segments_stationnaires, segmenter_points)


def _points(ts):
    return [(39.9, 116.3, 0.0, float(t)) for t in ts]


def test_segmenter_points_coupe_aux_trous_de_temps():
    segs = segmenter_points(_points([0, 1, 2, 700, 701]), seuil_temps=600)
    assert [len(s) for s in segs] == [3, 2]
    assert segmenter_points([], 600) == []


def test_segmenter_points_trou_egal_au_seuil_ne_coupe_pas():
    assert len(segmenter_points(_points([0, 600, 1200]), seuil_temps=600)) == 1


def test_filtrer_segments_par_taille():
    assert filtrer_segments([[1] * 5, [1] * 3], min_points=4) == [[1] * 5]


def test_filtrer_segments_stationnaires_en_km():
    t, lat, lon = marche(100, pas_m=1.0)                       # ~99 m parcourus
    immobile = pd.DataFrame({"lat": np.full(100, 39.9), "lon": np.full(100, 116.3)})
    mobile = pd.DataFrame({"lat": lat, "lon": lon})
    assert len(filtrer_segments_stationnaires([immobile, mobile], distance_min_km=0.05)) == 1
    assert len(filtrer_segments_stationnaires([immobile, mobile], distance_min_km=0.5)) == 0


# ---------------------------------------------------------------- coupure aux points marques (V5)
def _marche_avec_pic(n=400, indice_pic=200):
    rng = np.random.default_rng(2)
    dt = rng.uniform(1, 3, n)
    v = np.abs(1.4 + 0.2 * rng.standard_normal(n))
    v[indice_pic] = 60.0                                        # 216 km/h : artefact isole
    return segment_marque(v, dt)


def test_coupure_isole_le_pic_sans_raccorder_les_morceaux():
    df = _marche_avec_pic()
    marques = np.flatnonzero((df["type_suspect_vitesse"] == "artefact").to_numpy())
    assert len(marques) >= 1
    morceaux = couper_aux_points_marques(df, min_points=20)
    assert len(morceaux) == 2
    assert sum(len(m) for m in morceaux) == len(df) - len(marques)
    for m in morceaux:                                           # aucun point marque ne subsiste
        assert not (m["type_suspect_vitesse"] == "artefact").any()
    assert morceaux[0].index.max() < marques.min() < morceaux[1].index.min()   # le pic est entre les deux


def test_coupure_ne_coupe_jamais_aux_regimes():
    rng = np.random.default_rng(2)
    dt = rng.uniform(1, 3, 400)
    v = np.abs(1.4 + 0.2 * rng.standard_normal(400))
    v[150:260] *= 6                                              # regime soutenu (bus)
    df = segment_marque(v, dt)
    assert (df["type_suspect_vitesse"] == "regime").any()
    morceaux = couper_aux_points_marques(df, min_points=20)
    assert len(morceaux) == 1 and len(morceaux[0]) == len(df)


def test_coupure_sans_marque_renvoie_le_segment_tel_quel():
    df = segment_marque(np.full(300, 1.4), np.full(300, 2.0))
    out = couper_aux_points_marques(df, min_points=20)
    assert len(out) == 1 and out[0] is df
    assert couper_aux_points_marques(df, min_points=1000) == []


def test_coupure_applique_min_points_aux_morceaux():
    df = _marche_avec_pic(n=400, indice_pic=30)                  # un morceau de 30 points, un de ~370
    assert len(couper_aux_points_marques(df, min_points=20)) == 2
    assert len(couper_aux_points_marques(df, min_points=100)) == 1


def test_coupure_gere_les_colonnes_relues_depuis_csv():
    """Relu depuis un CSV, type_suspect_vitesse vaut NaN quand il est vide : ne doit pas planter."""
    df = _marche_avec_pic()
    df["type_suspect_vitesse"] = df["type_suspect_vitesse"].replace("", np.nan)
    assert len(couper_aux_points_marques(df, min_points=20)) == 2
