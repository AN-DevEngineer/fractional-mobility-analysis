import matplotlib.image as mpimg
import numpy as np

from conftest import segment_marque
from pretraitement.courbe import tracer_courbe_segment, tracer_courbe_serie


def _df():
    rng = np.random.default_rng(4)
    v = np.abs(1.4 + 0.2 * rng.standard_normal(300))
    v[100] = 60.0
    return segment_marque(v, rng.uniform(1, 3, 300))


def _dimensions(chemin):
    h, l = mpimg.imread(chemin).shape[:2]
    return l, h


def test_courbe_par_defaut_est_une_image_a_un_panneau(tmp_path):
    chemin = tracer_courbe_segment(_df(), 5, str(tmp_path / "log"), panneau_indice=False)
    assert chemin.endswith("courbe_distance_5.png")
    largeur, hauteur = _dimensions(chemin)
    assert largeur > 1000 and hauteur < 600


def test_option_deux_panneaux_donne_une_image_plus_haute(tmp_path):
    un = tracer_courbe_segment(_df(), 1, str(tmp_path / "a"), panneau_indice=False)
    deux = tracer_courbe_segment(_df(), 1, str(tmp_path / "b"), panneau_indice=True,
                                 h_estimes={"DFA": 0.8, "R/S": 0.7, "Whittle": None})
    assert _dimensions(deux)[1] > 1.4 * _dimensions(un)[1]


def test_segment_d_un_point_ne_trace_rien(tmp_path):
    assert tracer_courbe_segment(_df().iloc[:1], 0, str(tmp_path)) is None


def test_courbe_de_serie_pour_la_validation_a_l_aveugle(tmp_path):
    chemin = tracer_courbe_serie(np.arange(1.0, 101.0), np.abs(np.sin(np.arange(100))) * 5, str(tmp_path / "x" / "img.png"))
    assert chemin.endswith("img.png") and _dimensions(chemin)[0] > 1000
