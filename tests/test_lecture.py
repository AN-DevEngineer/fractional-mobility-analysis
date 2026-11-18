import os
import time
from datetime import datetime, timezone

import pytest

from conftest import T0, ecrire_plt, marche
from pretraitement.lecture import convertir_timestamp, lire_dossier_utilisateur, lire_fichier_plt


def test_convertir_timestamp_est_en_utc():
    attendu = datetime(2008, 10, 23, 2, 53, 4, tzinfo=timezone.utc).timestamp()
    assert convertir_timestamp("2008-10-23", "02:53:04") == pytest.approx(attendu)
    assert convertir_timestamp("2008-10-23", "02:53:04") == T0


def test_lire_fichier_plt_lit_les_points_et_ignore_les_lignes_defectueuses(tmp_path):
    t, lat, lon = marche(5)
    p = tmp_path / "a.plt"
    ecrire_plt(p, t, lat, lon, lignes_en_plus=["ligne,trop,courte", "", "39.9,116.3,0,100,40000.0,2008-99-99,25:61:00"])
    points = lire_fichier_plt(str(p))
    assert len(points) == 5
    lat0, lon0, alt0, t0 = points[0]
    assert (round(lat0, 6), round(lon0, 6), alt0, t0) == (round(lat[0], 6), 116.3, 100.0, T0)


def test_lire_fichier_plt_inexistant_renvoie_liste_vide(tmp_path):
    assert lire_fichier_plt(str(tmp_path / "absent.plt")) == []


def test_lire_dossier_utilisateur_trie_et_ignore_les_autres_fichiers(tmp_path):
    traj = tmp_path / "007" / "Trajectory"
    traj.mkdir(parents=True)
    for nom, t0 in (("b.plt", T0 + 1e5), ("a.plt", T0)):
        ecrire_plt(traj / nom, *marche(3, t0=t0))
    (traj / "notes.txt").write_text("pas un plt")
    trajets = lire_dossier_utilisateur(str(tmp_path / "007"))
    assert [t[0].split("/")[-1].split("\\")[-1] for t in trajets] == ["a.plt", "b.plt"]
    assert all(len(points) == 3 for _, points in trajets)


def test_lire_dossier_sans_trajectory_renvoie_liste_vide(tmp_path):
    assert lire_dossier_utilisateur(str(tmp_path)) == []


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset indisponible (Windows)")
@pytest.mark.parametrize("fuseau", ["Asia/Shanghai", "America/New_York"])
def test_convertir_timestamp_ne_depend_pas_du_fuseau_de_la_machine(fuseau):
    """La V4 utilisait l'heure locale : sur une machine hors UTC, tous les timestamps etaient decales."""
    ancien = os.environ.get("TZ")
    try:
        os.environ["TZ"] = fuseau
        time.tzset()
        assert convertir_timestamp("2008-10-23", "02:53:04") == T0
        assert convertir_timestamp("2008-07-01", "12:00:00") == 1214913600.0      # ete : heure d'ete aux Etats-Unis
    finally:
        if ancien is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = ancien
        time.tzset()
