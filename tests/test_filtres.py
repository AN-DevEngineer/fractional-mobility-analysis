"""Le marquage MAD de la Phase 3 : il OBSERVE, il ne retire jamais un point."""
import numpy as np

from conftest import segment_marque


def _marche(n=400, graine=2):
    rng = np.random.default_rng(graine)
    return np.abs(1.4 + 0.2 * rng.standard_normal(n)), rng.uniform(1, 3, n)


def test_marche_propre_n_a_aucun_point_suspect():
    v, dt = _marche()
    df = segment_marque(v, dt)
    assert int(df["suspect_vitesse"].sum()) == 0 and not bool(df["alerte_heterogene"].iloc[0])
    assert len(df) == 400                                       # aucun point retire


def test_pic_isole_est_marque_artefact():
    v, dt = _marche()
    v[200] = 60.0
    df = segment_marque(v, dt)
    idx = np.flatnonzero(df["suspect_vitesse"])
    assert 200 in idx and set(df["type_suspect_vitesse"][idx]) == {"artefact"}


def test_regime_soutenu_marque_seul_le_premier_point_et_alerte_le_segment():
    v, dt = _marche()
    v[150:260] *= 6
    df = segment_marque(v, dt)
    idx = np.flatnonzero(df["suspect_vitesse"])
    assert list(idx) == [150] and df["type_suspect_vitesse"][150] == "regime"
    assert bool(df["alerte_heterogene"].iloc[0])


def test_un_arret_n_est_jamais_un_artefact():
    """Une vitesse basse (arret) n'est pas candidate : le Z-score est asymetrique (correction 1 de seuil_dyn.pdf)."""
    v, dt = _marche()
    v[100:130] = 0.01
    assert int(segment_marque(v, dt)["suspect_vitesse"].sum()) == 0


def test_saut_de_distance_au_dela_du_seuil_absolu_est_marque_suspect_dist():
    v, dt = _marche()
    v[200] = 400.0                                              # ~800 m en 2 s : saut > MAX_DIST_M
    df = segment_marque(v, dt)
    assert df["suspect_dist"].any()
