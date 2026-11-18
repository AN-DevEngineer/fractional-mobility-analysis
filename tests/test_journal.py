import os

import numpy as np

from conftest import segment_marque
from pretraitement.journal import ecrire_log_utilisateur
from pretraitement.stats_segment import calculer_stats_segment


def _info(v, dt, retenu=True):
    df = segment_marque(v, dt)
    seg = [(float(r.lat), float(r.lon), 0.0, float(r.timestamp)) for r in df.itertuples()]
    return {"df": df, "fichier_source": "20081023025304.plt", "stats": calculer_stats_segment(seg), "retenu": retenu}


def test_un_fichier_log_par_segment(tmp_path):
    rng = np.random.default_rng(3)
    infos = [_info(np.full(120, 1.4), rng.uniform(1, 3, 120)), _info(np.full(150, 1.4), rng.uniform(1, 3, 150), retenu=False)]
    dossier = ecrire_log_utilisateur("007", infos, str(tmp_path))
    assert dossier == os.path.join(str(tmp_path), "007")
    assert sorted(os.listdir(dossier)) == ["log_segment_0.txt", "log_segment_1.txt"]
    texte = open(os.path.join(dossier, "log_segment_0.txt"), encoding="utf-8").read()
    assert "20081023025304.plt" in texte and len(texte) > 500


def test_un_gap_interne_est_signale(tmp_path):
    rng = np.random.default_rng(3)
    dt = rng.uniform(1, 3, 200)
    dt[100] = 120.0                                            # trou de 2 minutes dans le segment
    dossier = ecrire_log_utilisateur("007", [_info(np.full(200, 1.4), dt)], str(tmp_path))
    assert "GAP TEMPOREL" in open(os.path.join(dossier, "log_segment_0.txt"), encoding="utf-8").read()


def test_pas_de_gap_signale_sur_un_segment_regulier(tmp_path):
    dossier = ecrire_log_utilisateur("007", [_info(np.full(200, 1.4), np.full(200, 2.0))], str(tmp_path))
    assert "GAP TEMPOREL" not in open(os.path.join(dossier, "log_segment_0.txt"), encoding="utf-8").read()
