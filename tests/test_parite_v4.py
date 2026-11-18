"""
Parite numerique avec vos resultats REELS de la V4 (test d'integration).

Par defaut, utilise vos dossiers ./sortie et ./resultats_H a la racine du projet. S'ils sont vides, le
test est ignore. Pour pointer ailleurs :

    export M1_SORTIE=/chemin/sortie      (contient 000.csv, 001.csv, ...)
    export M1_RESULTATS=/chemin/resultats_H   (contient 000_H.csv, 001_H.csv, ...)
    python -m pytest tests/test_parite_v4.py

Verifie, segment par segment, que les versions rapides retrouvent h_dfa, h_rs et h_whittle ecrits par la V4.
"""
import os
from pathlib import Path

import pandas as pd
import pytest

from hurst.estimation_segment import series_du_segment
from hurst.dfa import dfa_h
from hurst.rs import rs_complet
from hurst.whittle import whittle_local

RACINE = Path(__file__).resolve().parents[1]
SORTIE = Path(os.environ.get("M1_SORTIE", RACINE / "sortie"))
RESULTATS = Path(os.environ.get("M1_RESULTATS", RACINE / "resultats_H"))


def _utilisateurs():
    if not (SORTIE.is_dir() and RESULTATS.is_dir()):
        return []
    return sorted(p.stem for p in SORTIE.glob("[0-9][0-9][0-9].csv") if (RESULTATS / f"{p.stem}_H.csv").exists())


@pytest.mark.skipif(not _utilisateurs(), reason="aucun couple sortie/<user>.csv + resultats_H/<user>_H.csv trouve")
@pytest.mark.parametrize("user", _utilisateurs())
def test_utilisateur_identique_a_la_v4(user):
    ref = pd.read_csv(RESULTATS / f"{user}_H.csv").set_index("segment_id")
    sortie = pd.read_csv(SORTIE / f"{user}.csv")
    verifies = 0
    for seg_id, g in sortie.groupby("segment_id", sort=True):
        if seg_id not in ref.index:
            continue
        d = series_du_segment(g[["lat", "lon", "timestamp"]].values.tolist())["d"]
        r = ref.loc[seg_id]
        assert dfa_h(d) == pytest.approx(r.h_dfa, abs=1e-6)
        assert rs_complet(d).h_brut == pytest.approx(r.h_rs, abs=1e-6)
        assert whittle_local(d).h == pytest.approx(r.h_whittle, abs=1e-9)
        verifies += 1
    assert verifies > 0
