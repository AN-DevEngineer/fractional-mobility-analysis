from pretraitement.nettoyage import nettoyer_points, point_valide, supprimer_doublons
from pretraitement.tri import trier_par_temps, verifier_ordre_temporel


def test_point_valide_bornes():
    assert point_valide((10.0, 10.0, 0.0, 1.0))
    assert not point_valide((95.0, 10.0, 0.0, 1.0))
    assert not point_valide((10.0, 190.0, 0.0, 1.0))
    assert not point_valide((10.0, 10.0, 0.0, 0.0))        # timestamp nul ou negatif


def test_nettoyer_points_retire_les_invalides():
    pts = [(10.0, 10.0, 0.0, 1.0), (95.0, 10.0, 0.0, 2.0), (10.0, 10.0, 0.0, 3.0)]
    assert nettoyer_points(pts) == [pts[0], pts[2]]


def test_supprimer_doublons_horodatage_consecutif():
    pts = [(1.0, 1.0, 0.0, 1.0), (1.1, 1.0, 0.0, 1.0), (1.2, 1.0, 0.0, 2.0)]
    assert supprimer_doublons(pts) == [pts[0], pts[2]]
    assert supprimer_doublons([]) == []


def test_trier_et_verifier_ordre():
    pts = [(0, 0, 0, 3.0), (0, 0, 0, 1.0), (0, 0, 0, 2.0)]
    assert verifier_ordre_temporel(pts) == [1]
    triee = trier_par_temps(pts)
    assert [p[3] for p in triee] == [1.0, 2.0, 3.0] and verifier_ordre_temporel(triee) == []
