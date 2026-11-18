import numpy as np
import pytest

from validation.monte_carlo import monte_carlo, resumer, tableau_latex_exactitude, taux_rejet_nul


def test_resume_et_tableau_latex_sont_coherents():
    mc = monte_carlo([0.5, 0.8], [200], 12, np.random.default_rng(0))
    assert len(mc) == 24 and {"dfa", "rs_brut", "rs_corrige", "whittle"} <= set(mc.columns)
    r = resumer(mc)
    assert len(r) == 2 * 4 and (r["rmse"] >= r["biais"].abs() - 1e-12).all()
    tex = tableau_latex_exactitude(r, "Test", "tab:test")
    assert tex.count("\\\\") >= 4 and "\\label{tab:test}" in tex and "R/S corrig" in tex


def test_taux_de_rejet_est_une_proportion():
    r = taux_rejet_nul(["iid", "fgn_0.9"], [300], 6, np.random.default_rng(1), ordres_ar=(1,))
    assert len(r) == 2 and r["taux_rejet"].between(0, 1).all()


@pytest.mark.lent
def test_exactitude_sur_fgn_dans_les_bornes_attendues():
    r = resumer(monte_carlo([0.7], [1000], 80, np.random.default_rng(2))).set_index("methode")
    assert abs(r.loc["dfa", "biais"]) < 0.03 and abs(r.loc["whittle", "biais"]) < 0.03
    assert r.loc["dfa", "rmse"] < 0.08


@pytest.mark.lent
def test_faux_positifs_controles_sur_memoire_courte():
    r = taux_rejet_nul(["ar1_0.5"], [500], 150, np.random.default_rng(3), ordres_ar=(5,))
    assert r["taux_rejet"].iloc[0] < 0.12
