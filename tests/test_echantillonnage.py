import numpy as np

from pretraitement.echantillonnage import profil_echantillonnage


def test_enregistrement_temporel(rng):
    dt = np.full(500, 5.0)
    d = rng.uniform(1, 12, 500)
    assert profil_echantillonnage(dt, d)["mode"] == "temps"


def test_enregistrement_par_distance(rng):
    d = rng.uniform(6.5, 7.5, 500)             # ~7 m a chaque point
    dt = rng.uniform(1, 12, 500)               # le temps varie avec la vitesse
    p = profil_echantillonnage(dt, d)
    assert p["mode"] == "distance" and p["part_d_regulieres"] > 0.9


def test_enregistrement_mixte(rng):
    d = rng.lognormal(1.0, 1.0, 500)
    dt = rng.uniform(1, 12, 500)
    assert profil_echantillonnage(dt, d)["mode"] == "mixte"


def test_entree_trop_courte_est_indeterminee():
    assert profil_echantillonnage(np.array([1.0]), np.array([2.0]))["mode"] == "indetermine"


def test_distance_dominee_par_le_pas_de_temps(rng):
    dt = rng.uniform(1, 30, 800)
    d = 1.4 * dt                                # marche a vitesse constante : d_i ne reflete que dt_i
    assert profil_echantillonnage(dt, d)["corr_d_dt"] > 0.99
