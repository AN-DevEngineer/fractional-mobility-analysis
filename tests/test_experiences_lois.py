"""
Tests de experiences/lois_de_puissance.py et de la partie reelle de experiences/mecanisme_on_off.py.

On fabrique des CSV de sortie/ synthetiques ou chaque utilisateur alterne arrets de duree de Pareto
et marche : on sait donc combien d'arrets existent. Les tests verifient la TUYAUTERIE (colonnes,
comptes, filtres, cas limites), pas la qualite statistique (voir test_clauset.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from experiences import lois_de_puissance as lp
from experiences import mecanisme_on_off as onoff_exp
from pretraitement.arrets import COLONNES_SEJOURS

M_PAR_DEGRE = 111_194.9
JEU = (100.0, 300.0)


def _ecrire_sortie_avec_arrets(dossier, utilisateurs=("000", "001"), nb_cycles=40, graine=3):
    """
    CSV au format de sortie/ : un seul segment continu par utilisateur, fait de `nb_cycles` cycles
    [arret de duree de Pareto >= 300 s (plafonnee a 3600 s) ; marche de 300 s]. Pas de temps : 10 s.
    """
    rng = np.random.default_rng(graine)
    Path(dossier).mkdir(parents=True, exist_ok=True)
    for u in utilisateurs:
        t, lat, lon = [], [], []
        temps, position = 1.2e9, 39.9
        for _ in range(nb_cycles):
            duree = min(300.0 * (1.0 - rng.random()) ** (-1.0 / 1.5), 3600.0)
            for _ in range(int(duree / 10.0)):
                t.append(temps); lat.append(position + rng.normal(0, 3) / M_PAR_DEGRE)
                lon.append(116.3 + rng.normal(0, 3) / 85_000); temps += 10.0
            for _ in range(30):
                position += 14.0 / M_PAR_DEGRE
                t.append(temps); lat.append(position + rng.normal(0, 3) / M_PAR_DEGRE)
                lon.append(116.3 + rng.normal(0, 3) / 85_000); temps += 10.0
        n = len(t)
        pd.DataFrame({"segment_id": 0, "point_id": np.arange(n), "lat": lat, "lon": lon, "alt": 0.0,
                      "timestamp": t}).to_csv(Path(dossier) / f"{u}.csv", index=False)
    return Path(dossier)


@pytest.fixture
def sortie(tmp_path):
    return _ecrire_sortie_avec_arrets(tmp_path / "sortie")


# ---------------------------------------------------------------- utilitaires
def test_nom_de_jeu_et_jeu_par_defaut():
    assert lp.nom_jeu(100.0, 300.0) == "100m_300s"
    assert lp.nom_jeu(200.0, 1200.0) == "200m_1200s"
    assert lp.jeu_par_defaut() == lp.nom_jeu(config.ARRET_RAYON_M, config.ARRET_DUREE_MIN_S)


def test_la_graine_ne_depend_que_de_la_cle():
    premier = lambda cle: lp._graine(cle).random()
    assert premier("a|b|c") == premier("a|b|c")
    assert premier("a|b|c") != premier("a|b|d")


def test_variable_inconnue_est_refusee():
    with pytest.raises(ValueError):
        lp._valeurs(pd.DataFrame(columns=COLONNES_SEJOURS), "inconnue")


# ---------------------------------------------------------------- collecte des sejours
def test_collecte_detecte_les_arrets_synthetiques(sortie):
    s = lp.collecter_sejours(str(sortie), [JEU])
    assert set(s["utilisateur"]) == {"000", "001"} and set(s["jeu"]) == {"100m_300s"}
    for u in ("000", "001"):
        n = (s["utilisateur"] == u).sum()
        assert 38 <= n <= 40                           # 40 cycles : le premier arret commence au 1er point (censure)


def test_collecte_peut_etre_restreinte_a_certains_utilisateurs(sortie):
    s = lp.collecter_sejours(str(sortie), [JEU], utilisateurs=["001"])
    assert set(s["utilisateur"]) == {"001"}


def test_collecte_ignore_les_fichiers_qui_ne_sont_pas_des_utilisateurs(sortie):
    (sortie / "notes.csv").write_text("a,b\n1,2\n")
    assert set(lp.collecter_sejours(str(sortie), [JEU])["utilisateur"]) == {"000", "001"}


def test_collecte_sur_dossier_vide(tmp_path):
    (tmp_path / "vide").mkdir()
    assert lp.collecter_sejours(str(tmp_path / "vide"), [JEU]).empty


def test_plusieurs_jeux_donnent_plusieurs_lots(sortie):
    s = lp.collecter_sejours(str(sortie), [(100.0, 300.0), (200.0, 1200.0)])
    assert set(s["jeu"]) == {"100m_300s", "200m_1200s"}
    n_court = (s["jeu"] == "100m_300s").sum()
    n_long = (s["jeu"] == "200m_1200s").sum()
    assert n_long < n_court                            # un seuil de duree plus exigeant detecte moins d'arrets


# ---------------------------------------------------------------- resume des distributions
def test_une_ligne_par_analyse_et_colonnes_attendues(sortie):
    s = lp.collecter_sejours(str(sortie), [JEU])
    r = lp.resumer_distributions(s, nb_bootstrap=5, nb_bootstrap_utilisateur=0, n_min_utilisateur=30,
                                 un_seul_jeu=True, avec_alternatives=False)
    # population agregee : 4 variables ; chaque utilisateur (>= 30 observations) : 4 variables
    assert len(r) == 4 + 2 * 4
    assert set(r["population"]) == {"agrege", "000", "001"}
    assert set(r["variable"]) == {"sejour", "distance", "sejour_avec_censurees", "distance_a_cheval"}
    for col in ("jeu", "population", "variable", "n_observations", "alpha", "xmin", "verdict", "dans_domaine_on_off",
                "empreinte_config"):
        assert col in r.columns
    assert (r["empreinte_config"] == config.empreinte_config()).all()


def test_un_utilisateur_avec_trop_peu_d_observations_est_ignore(sortie):
    s = lp.collecter_sejours(str(sortie), [JEU])
    r = lp.resumer_distributions(s, nb_bootstrap=0, nb_bootstrap_utilisateur=0, n_min_utilisateur=10_000,
                                 un_seul_jeu=True, avec_alternatives=False)
    assert set(r["population"]) == {"agrege"}


def test_les_variantes_de_sensibilite_ne_concernent_que_le_jeu_par_defaut(sortie):
    s = lp.collecter_sejours(str(sortie), [JEU, (200.0, 1200.0)])
    r = lp.resumer_distributions(s, nb_bootstrap=0, nb_bootstrap_utilisateur=0, n_min_utilisateur=10_000,
                                 avec_alternatives=False)
    autre = r[(r["jeu"] == "200m_1200s")]
    assert set(autre["variable"]) == {"sejour", "distance"}


def test_resume_sans_sejours_est_vide():
    assert lp.resumer_distributions(pd.DataFrame(), 0, 0).empty


# ---------------------------------------------------------------- partie reelle du lien ON/OFF
def _ecrire_lois(chemin, lignes):
    colonnes = dict(jeu=lp.jeu_par_defaut(), variable="sejour", xmin=300.0, p_bootstrap=0.5)
    pd.DataFrame([dict(colonnes, **l) for l in lignes]).to_csv(chemin, index=False)


def _ecrire_h(dossier, utilisateur, h_dfa=(0.8, 0.85, 0.9), n_points=(600, 700, 800)):
    pd.DataFrame({"segment_id": range(len(h_dfa)), "n_points": n_points, "h_dfa": h_dfa,
                  "pas_median_s": 5.0, "dfa_sup_1": False}).to_csv(Path(dossier) / f"{utilisateur}_H.csv", index=False)


def test_partie_reelle_sans_fichier_de_lois(tmp_path, capsys):
    r = onoff_exp.partie_reelle(str(tmp_path / "absent.csv"), str(tmp_path))
    assert r.empty and "introuvable" in capsys.readouterr().out


def test_partie_reelle_hors_domaine_rapporte_que_la_theorie_ne_s_applique_pas(tmp_path, capsys):
    """Exposants de densite < 2 : exposant de survie < 1, hors du domaine de H = (3 - a) / 2."""
    lois = tmp_path / "lois.csv"
    _ecrire_lois(lois, [dict(population=p, alpha=1.6, dans_domaine_on_off=False, verdict="plausible", n_observations=50)
                        for p in ("000", "001", "002")])
    r = onoff_exp.partie_reelle(str(lois), str(tmp_path), nb_rep=2)
    sortie = capsys.readouterr().out
    assert r.empty and "ne s'applique pas" in sortie and "0  (0%)" in sortie


def test_partie_reelle_dans_le_domaine_compare_h_observe_a_la_theorie(tmp_path):
    lois = tmp_path / "lois.csv"
    _ecrire_lois(lois, [dict(population="000", alpha=2.5, dans_domaine_on_off=True, verdict="plausible", n_observations=60)])
    _ecrire_h(tmp_path, "000")
    r = onoff_exp.partie_reelle(str(lois), str(tmp_path), nb_rep=3)
    assert len(r) == 1
    ligne = r.iloc[0]
    assert ligne["alpha_survie"] == pytest.approx(1.5)
    assert ligne["h_theorique"] == pytest.approx(0.75)                    # (3 - 1,5) / 2
    assert ligne["h_observe"] == pytest.approx(0.85)                      # mediane de h_dfa
    assert ligne["ecart_obs_theorie"] == pytest.approx(0.85 - 0.75)
    assert ligne["n_serie"] == 699                                         # mediane de n_points (700) moins 1


def test_partie_reelle_ignore_les_utilisateurs_dont_la_loi_n_est_pas_plausible(tmp_path):
    lois = tmp_path / "lois.csv"
    _ecrire_lois(lois, [dict(population="000", alpha=2.5, dans_domaine_on_off=True, verdict="rejetee", n_observations=60)])
    _ecrire_h(tmp_path, "000")
    assert onoff_exp.partie_reelle(str(lois), str(tmp_path), nb_rep=2).empty


def test_partie_reelle_sans_fichier_h_ne_plante_pas(tmp_path):
    lois = tmp_path / "lois.csv"
    _ecrire_lois(lois, [dict(population="000", alpha=2.5, dans_domaine_on_off=True, verdict="plausible", n_observations=60)])
    assert onoff_exp.partie_reelle(str(lois), str(tmp_path), nb_rep=2).empty    # resultats_H/000_H.csv absent
