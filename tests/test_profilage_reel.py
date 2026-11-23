import os

import numpy as np
import pandas as pd

from outils.labels_geolife import lire_labels, mode_du_segment, MAPPING_5
from outils.chargement_segments import utilisateurs, segments_utilisateur
from profilage.simulation_modes import generer_segment


def _ecrire_utilisateur(dossier, geo, user, t0):
    s = generer_segment("bus", 400, np.random.default_rng(0))
    t = t0 + 5.0 * np.arange(401)
    pd.DataFrame(dict(segment_id=0, point_id=np.arange(401), lat=s["lat"], lon=s["lon"], alt=0.0, timestamp=t,
                      suspect_dist=False, suspect_vitesse=False)).to_csv(os.path.join(dossier, f"{user}.csv"), index=False)
    os.makedirs(os.path.join(geo, user), exist_ok=True)
    with open(os.path.join(geo, user, "labels.txt"), "w") as f:
        f.write("Start Time\tEnd Time\tTransportation Mode\n")
        f.write("2008/01/01 00:00:00\t2008/01/01 00:40:00\tbus\n")
    return t


def test_chargement_et_etiquetage(tmp_path):
    sortie, geo = str(tmp_path / "sortie"), str(tmp_path / "geo")
    os.makedirs(sortie)
    t0 = 1199145600.0            # 2008-01-01 00:00:00 UTC
    t = _ecrire_utilisateur(sortie, geo, "000", t0)
    assert utilisateurs(sortie) == ["000"]
    seg = next(segments_utilisateur("000", sortie))
    assert len(seg["d"]) == 400 and seg["points"].shape == (401, 3)
    labels = lire_labels(os.path.join(geo, "000", "labels.txt"))
    assert mode_du_segment(t[0], t[-1], labels) == "bus"
    assert MAPPING_5["bus"] == "bus"
    assert mode_du_segment(t[0] + 5000, t[-1] + 5000, labels) is None        # segment hors etiquette
    assert mode_du_segment(t[0], t[-1] + 4000, labels, seuil=0.8) is None    # couverture partielle < 80 %


def test_labels_absents():
    assert lire_labels("/chemin/inexistant/labels.txt") == []
