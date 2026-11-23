import os

import numpy as np
import pandas as pd

from outils.chargement_segments import segments_utilisateur
from outils.exporter_pour_analyse import exporter_resume, exporter_series, lire_series_npz
from profilage.simulation_modes import generer_segment


def _faux_projet(tmp_path):
    sortie, res = tmp_path / "sortie", tmp_path / "resultats_H"
    sortie.mkdir(); res.mkdir()
    rng = np.random.default_rng(0)
    for u, classe in (("000", "bus"), ("001", "train")):
        blocs = []
        for sid in range(3):
            s = generer_segment(classe, 350 + 40 * sid, rng)
            n = len(s["lat"])
            blocs.append(pd.DataFrame(dict(segment_id=sid, point_id=np.arange(n), lat=s["lat"], lon=s["lon"], alt=0.0,
                                           timestamp=1.2e9 + 5.0 * np.arange(n), suspect_dist=False, suspect_vitesse=False)))
        pd.concat(blocs).to_csv(sortie / f"{u}.csv", index=False)
        pd.DataFrame(dict(segment_id=range(3), n_points=[350, 390, 430], h_dfa=[.7, .8, .9], h_rs=[.7, .7, .8],
                          h_rs_corrige=[.64, .65, .74], h_whittle=[.7, .75, .8])).to_csv(res / f"{u}_H.csv", index=False)
    return str(sortie), str(res)


def test_export_resume(tmp_path):
    sortie, res = _faux_projet(tmp_path)
    exporter_resume(res, str(tmp_path / "export"))
    df = pd.read_csv(tmp_path / "export" / "resume_H.csv.gz", dtype={"utilisateur": str})
    assert len(df) == 6 and set(df.utilisateur) == {"000", "001"} and "h_rs_corrige" in df.columns


def test_export_series_identique_a_la_lecture_directe(tmp_path):
    sortie, res = _faux_projet(tmp_path)
    exp = str(tmp_path / "export")
    exporter_series(sortie, exp, mo_max=0.002)            # tres petit : force plusieurs fichiers
    assert len([f for f in os.listdir(exp) if f.startswith("series_")]) >= 2
    reference = [s for u in ("000", "001") for s in segments_utilisateur(u, sortie)]
    lus = list(lire_series_npz(exp))
    assert len(lus) == len(reference) == 6
    for a, b in zip(lus, reference):
        assert a["utilisateur"] == b["utilisateur"] and a["segment_id"] == b["segment_id"]
        assert np.allclose(a["d"], b["d"], rtol=1e-6)
