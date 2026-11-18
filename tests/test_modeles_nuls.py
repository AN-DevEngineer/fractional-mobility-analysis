import numpy as np
import pytest

from simulation.bruit_fractionnaire import bruit_gaussien_fractionnaire
from simulation.marginale_lognormale import rendre_positive
from simulation.memoire_courte import ar1
from validation.modeles_nuls import ajuster_ar, gaussianiser, surrogats, evaluer_depassement_dfa


def test_les_surrogats_conservent_exactement_la_marginale(rng):
    x = rendre_positive(ar1(500, 0.6, rng), 1.0)
    for modele in ("ar", "melange"):
        S = surrogats(x, rng, modele=modele, nb=7)
        assert S.shape == (7, 500)
        assert all(np.allclose(np.sort(s), np.sort(x)) for s in S)


def test_ajustement_ar_retrouve_le_coefficient(rng):
    phi, s2 = ajuster_ar(gaussianiser(ar1(20000, 0.6, rng)), 5)
    assert phi[0] == pytest.approx(0.6, abs=0.05) and s2 > 0


def test_ajustement_ar_choisit_un_ordre_faible_sur_bruit_blanc(rng):
    phi, _ = ajuster_ar(rng.standard_normal(5000), 5)
    assert len(phi) <= 1


def test_pvaleurs_du_melange_sont_a_peu_pres_uniformes(rng):
    p = [evaluer_depassement_dfa(rng.standard_normal(600), rng, modele="melange")["p"] for _ in range(80)]
    assert np.mean(p) == pytest.approx(0.5, abs=0.1)
    assert np.mean(np.array(p) < 0.05) < 0.12


def test_detecte_une_forte_memoire_longue_contre_un_nul_i_i_d(rng):
    p = [evaluer_depassement_dfa(bruit_gaussien_fractionnaire(1000, 0.9, rng), rng, modele="melange")["p"] for _ in range(30)]
    assert np.mean(np.array(p) < 0.05) > 0.9


def test_bandes_de_confiance_coherentes(rng):
    out = evaluer_depassement_dfa(bruit_gaussien_fractionnaire(1500, 0.7, rng), rng, nb_surrogats=30, avec_bandes=True)
    k = len(out["echelles_locales"])
    assert k > 3 and len(out["locales_obs"]) == len(out["locales_bas"]) == len(out["locales_haut"]) == k
    assert np.all(out["locales_bas"] <= out["locales_haut"])


def test_serie_courte_donne_nan(rng):
    assert np.isnan(evaluer_depassement_dfa(rng.standard_normal(10), rng)["p"])


def test_modele_inconnu_est_refuse(rng):
    with pytest.raises(ValueError):
        surrogats(rng.standard_normal(50), rng, modele="inconnu")


def test_le_resultat_ne_depend_que_de_la_graine():
    x = bruit_gaussien_fractionnaire(400, 0.7, np.random.default_rng(1))
    a = evaluer_depassement_dfa(x, np.random.default_rng(9))
    b = evaluer_depassement_dfa(x, np.random.default_rng(9))
    assert a == b
