import numpy as np
import pandas as pd
import pytest

from conftest import ecrire_sortie
from experiences import mecanisme_on_off, sensibilite_pretraitement as sens, validation_visuelle as vis
from hurst.dfa import dfa_h
from hurst.estimation_segment import series_du_segment


@pytest.fixture
def sortie(tmp_path):
    return ecrire_sortie(tmp_path / "sortie")


# ---------------------------------------------------------------- sensibilite_pretraitement
def test_sensibilite_donne_une_ligne_par_segment_et_variante(sortie):
    df = sens.analyser(str(sortie))
    assert len(df) == 2 * 4 * len(sens.VARIANTES)
    assert set(df["variante"]) == set(sens.VARIANTES)


def test_la_variante_de_base_est_le_h_de_la_v4(sortie):
    df = sens.analyser(str(sortie))
    seg = sens.lire_sortie(str(sortie / "000.csv"))[1]
    d = series_du_segment(seg[["lat", "lon", "timestamp"]].to_numpy(float))["d"]
    ligne = df[(df.utilisateur == "000") & (df.segment == 1) & (df.variante == "base")].iloc[0]
    assert ligne["h_dfa"] == pytest.approx(dfa_h(d))


def test_coupure_aux_artefacts_ne_change_que_les_segments_avec_artefact(sortie):
    df = sens.analyser(str(sortie))
    piv = df.pivot_table(index=["utilisateur", "segment"], columns="variante", values="h_dfa")
    sans_artefact = piv.xs(1, level="segment")           # segments impairs : aucun artefact injecte
    assert np.allclose(sans_artefact["coupure_artefacts"], sans_artefact["base"])
    avec = piv.xs(0, level="segment")
    assert not np.allclose(avec["coupure_artefacts"], avec["base"])


def test_placebo_retire_autant_de_points_que_la_coupure_aux_artefacts(sortie):
    seg = sens.lire_sortie(str(sortie / "000.csv"))[0]
    reel = sens.morceaux_de_la_variante(seg, "coupure_artefacts")
    placebo = sens.morceaux_de_la_variante(seg, "coupure_placebo")
    assert sum(len(m) for m in placebo) == sum(len(m) for m in reel)
    assert placebo[0].index.tolist() == sens.morceaux_de_la_variante(seg, "coupure_placebo")[0].index.tolist()   # reproductible


def test_coupure_aux_intervalles_ne_raccorde_pas():
    df = pd.DataFrame({"timestamp": list(range(0, 100)) + list(range(400, 500)), "lat": 0.0, "lon": 0.0})
    morceaux = sens.couper_aux_intervalles(df, seuil_s=60.0, min_points=10)
    assert [len(m) for m in morceaux] == [100, 100]


def test_resume_compare_chaque_variante_a_la_base(sortie):
    resume = sens.resumer(sens.analyser(str(sortie)))
    base = resume[(resume.variante == "base") & (resume.methode == "h_dfa")].iloc[0]
    assert base["ecart_abs_median_vs_base"] == 0 and base["part_ecart_sup_0_1"] == 0
    assert {"h_dfa", "h_whittle"} == set(resume["methode"])


# ---------------------------------------------------------------- validation_visuelle
def test_preparer_fabrique_images_cle_et_feuille(sortie, tmp_path):
    dossier = tmp_path / "visuel"
    cle = vis.preparer(str(sortie), str(dossier), n_reels=6, n_leurres=4, graine=1)
    assert len(cle) == 10 and sorted(cle["type"].unique()) == ["ar", "melange", "reel"]
    assert (cle["type"] == "reel").sum() == 6
    assert len(list((dossier / "images").glob("img_*.png"))) == 10
    feuille = pd.read_csv(dossier / "feuille_reponses.csv")
    assert list(feuille["code"]) == list(cle["code"]) and feuille["h_visuel"].isna().all()
    assert (dossier / "LISEZ_MOI.txt").exists()


def test_les_images_ne_revelent_rien_dans_leur_nom(sortie, tmp_path):
    cle = vis.preparer(str(sortie), str(tmp_path / "v"), n_reels=6, n_leurres=4, graine=1)
    assert all(c.startswith("img_") for c in cle["code"])
    assert list(cle["type"]) != sorted(cle["type"])        # l'ordre est melange : le type ne se lit pas dans l'ordre


def test_evaluer_retrouve_un_accord_parfait_et_distingue_les_leurres(sortie, tmp_path):
    dossier = tmp_path / "v"
    cle = vis.preparer(str(sortie), str(dossier), n_reels=8, n_leurres=4, graine=2)
    feuille = pd.DataFrame({"code": cle["code"], "h_visuel": np.where(cle["type"] == "reel", cle["h_dfa"], 0.1), "commentaire": ""})
    feuille.to_csv(dossier / "feuille_reponses.csv", index=False)
    r = vis.evaluer(str(dossier)).set_index("rubrique")["valeur"]
    assert r["Spearman visuel vs h_dfa (reel)"] == pytest.approx(1.0)
    assert r["erreur absolue moyenne vs h_dfa (reel)"] == pytest.approx(0.0, abs=1e-12)
    assert r["AUC reel vs leurres (0.5 = indiscernable)"] == pytest.approx(1.0)    # leurres tous plus bas que tous les reels
    assert r["h_visuel moyen (melange)"] == pytest.approx(0.1)


def test_evaluer_refuse_une_feuille_vide(sortie, tmp_path):
    vis.preparer(str(sortie), str(tmp_path / "v"), n_reels=6, n_leurres=2, graine=3)
    with pytest.raises(ValueError, match="remplissez"):
        vis.evaluer(str(tmp_path / "v"))


def test_preparer_refuse_trop_de_segments_demandes(sortie, tmp_path):
    with pytest.raises(ValueError):
        vis.preparer(str(sortie), str(tmp_path / "v"), n_reels=500, n_leurres=0)


# ---------------------------------------------------------------- mecanisme_on_off
def test_theorie_on_off_donne_les_colonnes_attendues():
    df = mecanisme_on_off.verifier_theorie(alphas=(1.4, 1.8), n=800, nb_rep=8, rng=np.random.default_rng(1))
    assert len(df) == 2 and df["h_theorique"].tolist() == pytest.approx([0.8, 0.6])
    assert {"dfa_biais", "whittle_biais", "rs_corrige_biais"} <= set(df.columns)
