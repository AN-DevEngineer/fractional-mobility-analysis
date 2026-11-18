"""
Tests de pretraitement/arrets.py : detection d'arrets, sejours, distances entre arrets.

Les trajectoires sont SYNTHETIQUES : on sait d'avance combien d'arrets il y a et combien ils durent.
Pas de temps : 10 s. Marche a 1,4 m/s = 14 m par pas, donc il faut 8 pas (112 m) pour sortir d'un
rayon de 100 m : un arret de duree D est mesure D + environ 80 s (instant ou on le voit ressortir).
"""
import numpy as np
import pandas as pd
import pytest

from pretraitement.arrets import (COLONNES_DISTANCES, COLONNES_SEJOURS, detecter_arrets, distances_entre_arrets,
                                  lire_points_sortie, sejours_utilisateur, temps_de_sejour)

M_PAR_DEGRE = 111_194.9
T0 = 1.2e9


def _trajet(phases, pas_s=10.0, vitesse_ms=1.4, graine=0):
    """
    Trajet synthetique construit a partir de phases [("marche", duree_s), ("arret", duree_s), ...].
    Marche : progression vers le nord. Arret : position fixe avec un bruit GPS d'environ 3 m.
    Renvoie (t, lat, lon).
    """
    rng = np.random.default_rng(graine)
    t, lat, lon = [], [], []
    temps, position = T0, 39.9
    for type_, duree in phases:
        for _ in range(int(duree / pas_s)):
            if type_ == "marche":
                position += vitesse_ms * pas_s / M_PAR_DEGRE
            t.append(temps)
            lat.append(position + rng.normal(0, 3) / M_PAR_DEGRE)
            lon.append(116.3 + rng.normal(0, 3) / 85_000)
            temps += pas_s
    return np.array(t), np.array(lat), np.array(lon)


# ---------------------------------------------------------------- detecter_arrets
def test_detecte_un_arret_de_15_minutes_entoure_de_marche():
    t, lat, lon = _trajet([("marche", 600), ("arret", 900), ("marche", 600)])
    arrets = detecter_arrets(t, lat, lon)
    assert len(arrets) == 1
    a = arrets[0]
    assert 900 <= a["duree_s"] <= 1100
    assert not a["censuree_debut"] and not a["censuree_fin"]


def test_detecte_plusieurs_arrets_successifs():
    phases = [("marche", 300)]
    for _ in range(5):
        phases += [("arret", 600), ("marche", 300)]
    arrets = detecter_arrets(*_trajet(phases))
    assert len(arrets) == 5
    assert all(600 <= a["duree_s"] <= 800 for a in arrets)
    assert all(arrets[i]["fin"] <= arrets[i + 1]["debut"] for i in range(4))     # dans l'ordre du temps, sans recouvrement


def test_ignore_un_arret_plus_court_que_la_duree_minimale():
    """Un arret de 150 s est mesure ~230 s (arret + ~80 s pour sortir du rayon) : sous le seuil de 300 s."""
    t, lat, lon = _trajet([("marche", 600), ("arret", 150), ("marche", 600)])
    assert detecter_arrets(t, lat, lon) == []


def test_la_duree_minimale_est_un_parametre():
    t, lat, lon = _trajet([("marche", 600), ("arret", 150), ("marche", 600)])
    assert len(detecter_arrets(t, lat, lon, duree_min_s=120)) == 1


def test_le_rayon_est_un_parametre():
    """Un va-et-vient de ~150 m n'est un arret qu'avec un rayon assez grand."""
    rng = np.random.default_rng(1)
    t = T0 + 10.0 * np.arange(120)                                  # 20 minutes
    lat = 39.9 + 75.0 * np.sin(np.arange(120) / 5.0) / M_PAR_DEGRE   # oscille de +/- 75 m
    lon = np.full(120, 116.3) + rng.normal(0, 1, 120) / 85_000
    assert len(detecter_arrets(t, lat, lon, rayon_m=50.0, inclure_dernier=True)) == 0
    assert len(detecter_arrets(t, lat, lon, rayon_m=200.0, inclure_dernier=True)) == 1


def test_arret_jamais_termine_detecte_seulement_avec_inclure_dernier():
    t, lat, lon = _trajet([("marche", 600), ("arret", 900)])
    assert detecter_arrets(t, lat, lon) == []
    arrets = detecter_arrets(t, lat, lon, inclure_dernier=True)
    assert len(arrets) == 1 and arrets[0]["censuree_fin"]


def test_arret_qui_commence_au_premier_point_est_censure_a_gauche():
    t, lat, lon = _trajet([("arret", 900), ("marche", 600)])
    arrets = detecter_arrets(t, lat, lon)
    assert len(arrets) == 1 and arrets[0]["censuree_debut"] and not arrets[0]["censuree_fin"]


def test_un_trou_d_enregistrement_coupe_la_serie_au_lieu_de_la_raccorder():
    """Deux morceaux immobiles de 190 s chacun, separes par un trou de 2000 s : jamais un arret de 2380 s."""
    t = np.concatenate([T0 + 10.0 * np.arange(20), T0 + 2000 + 190 + 10.0 * np.arange(20)])
    lat = np.full(40, 39.9)
    lon = np.full(40, 116.3)
    assert detecter_arrets(t, lat, lon, trou_max_s=900.0, inclure_dernier=True) == []
    raccorde = detecter_arrets(t, lat, lon, trou_max_s=1e9, inclure_dernier=True)
    assert len(raccorde) == 1 and raccorde[0]["duree_s"] > 2000      # c'est exactement l'erreur qu'on evite


def test_position_de_l_arret_est_le_centroide_des_points():
    t, lat, lon = _trajet([("marche", 600), ("arret", 900), ("marche", 600)])
    a = detecter_arrets(t, lat, lon)[0]
    dans_l_arret = (t >= a["debut"]) & (t < a["fin"])
    assert a["lat"] == pytest.approx(lat[dans_l_arret].mean())
    assert a["lon"] == pytest.approx(lon[dans_l_arret].mean())
    assert a["n_points"] == dans_l_arret.sum()


def test_entrees_invalides():
    with pytest.raises(ValueError):
        detecter_arrets([1.0, 2.0], [1.0], [1.0, 2.0])
    assert detecter_arrets([1.0], [1.0], [1.0]) == []
    assert detecter_arrets([], [], []) == []


# ---------------------------------------------------------------- sejours_utilisateur
def _points(segments):
    """DataFrame au format de sortie/ a partir de {segment_id: (t, lat, lon)}."""
    parties = [pd.DataFrame({"segment_id": sid, "timestamp": t, "lat": lat, "lon": lon})
               for sid, (t, lat, lon) in segments.items()]
    return pd.concat(parties, ignore_index=True)


def test_sejours_internes_ont_les_colonnes_attendues():
    t, lat, lon = _trajet([("marche", 600), ("arret", 900), ("marche", 600)])
    s = sejours_utilisateur(_points({0: (t, lat, lon)}), utilisateur="007")
    assert list(s.columns) == COLONNES_SEJOURS
    assert len(s) == 1 and s.loc[0, "type"] == "interne" and s.loc[0, "utilisateur"] == "007"
    assert not s.loc[0, "censuree"]


def _segment_apres(t0, lat0, lon0, trou_s, ecart_m):
    """
    Second segment de marche (29 points) qui reprend `trou_s` secondes apres la fin du premier,
    a `ecart_m` metres (vers le nord) du DERNIER point du premier.
    """
    n = 29
    t1 = t0[-1] + trou_s + 10.0 * np.arange(n)
    lat1 = lat0[-1] + ecart_m / M_PAR_DEGRE + 14.0 * np.arange(n) / M_PAR_DEGRE
    return t1, lat1, np.full(n, lon0[-1])


def test_pause_entre_deux_segments_proches_est_un_sejour_censure():
    """Appareil eteint 2 h puis rallume a 30 m du point d'extinction : pause de 7200 s, toujours censuree."""
    t0, lat0, lon0 = _trajet([("marche", 290)])
    t1, lat1, lon1 = _segment_apres(t0, lat0, lon0, trou_s=7200.0, ecart_m=30.0)
    s = sejours_utilisateur(_points({0: (t0, lat0, lon0), 1: (t1, lat1, lon1)}))
    entre = s[s["type"] == "entre_segments"]
    assert len(entre) == 1
    assert entre["duree_s"].iloc[0] == pytest.approx(7200.0)
    assert bool(entre["censuree"].iloc[0])


def test_pas_de_pause_si_le_second_segment_reprend_loin():
    t0, lat0, lon0 = _trajet([("marche", 290)])
    t1, lat1, lon1 = _segment_apres(t0, lat0, lon0, trou_s=7200.0, ecart_m=5000.0)
    s = sejours_utilisateur(_points({0: (t0, lat0, lon0), 1: (t1, lat1, lon1)}))
    assert (s["type"] == "entre_segments").sum() == 0


def test_pas_de_pause_si_le_trou_est_plus_court_que_la_duree_minimale():
    t0, lat0, lon0 = _trajet([("marche", 290)])
    t1, lat1, lon1 = _segment_apres(t0, lat0, lon0, trou_s=60.0, ecart_m=10.0)
    s = sejours_utilisateur(_points({0: (t0, lat0, lon0), 1: (t1, lat1, lon1)}))
    assert len(s) == 0


def test_sejours_utilisateur_refuse_un_dataframe_incomplet():
    with pytest.raises(ValueError, match="manquantes"):
        sejours_utilisateur(pd.DataFrame({"segment_id": [0], "timestamp": [1.0]}))


def test_sejours_utilisateur_sur_dataframe_vide():
    s = sejours_utilisateur(pd.DataFrame({"segment_id": [], "timestamp": [], "lat": [], "lon": []}))
    assert len(s) == 0 and list(s.columns) == COLONNES_SEJOURS


# ---------------------------------------------------------------- temps_de_sejour
def _sejours_manuels():
    colonnes = dict(utilisateur="u", lat=39.9, lon=116.3, n_points=10)
    return pd.DataFrame([
        dict(colonnes, segment_id=0, type="interne", debut=0, fin=600, duree_s=600.0, censuree=False),
        dict(colonnes, segment_id=0, type="interne", debut=1000, fin=1800, duree_s=800.0, censuree=True),
        dict(colonnes, segment_id=0, type="entre_segments", debut=2000, fin=9200, duree_s=7200.0, censuree=True),
    ])[COLONNES_SEJOURS]


def test_temps_de_sejour_exclut_par_defaut_les_censures_et_les_pauses():
    assert list(temps_de_sejour(_sejours_manuels())) == [600.0]


def test_temps_de_sejour_peut_tout_garder_pour_la_sensibilite():
    assert sorted(temps_de_sejour(_sejours_manuels(), seulement_non_censurees=False)) == [600.0, 800.0, 7200.0]


def test_temps_de_sejour_sans_sejour():
    assert len(temps_de_sejour(pd.DataFrame(columns=COLONNES_SEJOURS))) == 0


# ---------------------------------------------------------------- distances_entre_arrets
def _arrets_en_ligne(decalages_m, segments, types=None):
    """Arrets alignes vers le nord a des distances cumulees donnees (m)."""
    types = types or ["interne"] * len(decalages_m)
    return pd.DataFrame([
        dict(utilisateur="u", segment_id=seg, type=ty, debut=1000.0 * k, fin=1000.0 * k + 600, duree_s=600.0,
             lat=39.9 + d / M_PAR_DEGRE, lon=116.3, n_points=10, censuree=False)
        for k, (d, seg, ty) in enumerate(zip(decalages_m, segments, types))])[COLONNES_SEJOURS]


def test_distances_entre_arrets_consecutifs():
    d = distances_entre_arrets(_arrets_en_ligne([0, 1000, 3000], [0, 0, 0]))
    assert list(d.columns) == COLONNES_DISTANCES
    assert d["distance_m"].tolist() == pytest.approx([1000.0, 2000.0], rel=1e-3)
    assert d["meme_segment"].all()


def test_distances_a_cheval_sur_deux_segments_exclues_par_defaut():
    s = _arrets_en_ligne([0, 1000], [0, 1])
    assert len(distances_entre_arrets(s)) == 0
    a_cheval = distances_entre_arrets(s, meme_segment_seulement=False)
    assert len(a_cheval) == 1 and not a_cheval["meme_segment"].iloc[0]


def test_distances_sous_le_seuil_minimal_ecartees():
    d = distances_entre_arrets(_arrets_en_ligne([0, 50, 1050], [0, 0, 0]), distance_min_m=100.0)
    assert d["distance_m"].tolist() == pytest.approx([1000.0], rel=1e-3)


def test_une_pause_entre_segments_n_est_jamais_dans_le_meme_segment():
    s = _arrets_en_ligne([0, 1000], [0, 0], types=["interne", "entre_segments"])
    assert len(distances_entre_arrets(s)) == 0


def test_distances_sans_assez_d_arrets():
    assert len(distances_entre_arrets(_arrets_en_ligne([0], [0]))) == 0
    assert len(distances_entre_arrets(pd.DataFrame(columns=COLONNES_SEJOURS))) == 0


def test_les_distances_ne_melangent_pas_les_utilisateurs():
    a = _arrets_en_ligne([0, 1000], [0, 0])
    b = _arrets_en_ligne([5000, 7000], [0, 0]).assign(utilisateur="v")
    d = distances_entre_arrets(pd.concat([a, b], ignore_index=True))
    assert sorted(d["distance_m"].round(0).tolist()) == [1000.0, 2000.0]     # pas de paire u -> v


# ---------------------------------------------------------------- lecture
def test_lire_points_sortie_garde_quatre_colonnes_et_trie(tmp_path):
    chemin = tmp_path / "007.csv"
    pd.DataFrame({"segment_id": [1, 0, 0], "point_id": [0, 1, 0], "lat": [3.0, 2.0, 1.0],
                  "lon": [0.0, 0.0, 0.0], "alt": [0, 0, 0], "timestamp": [50.0, 20.0, 10.0]}).to_csv(chemin, index=False)
    p = lire_points_sortie(chemin)
    assert list(p.columns) == ["segment_id", "timestamp", "lat", "lon"]
    assert p["timestamp"].tolist() == [10.0, 20.0, 50.0] and p["segment_id"].tolist() == [0, 0, 1]
