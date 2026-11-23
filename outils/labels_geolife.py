# ./outils/labels_geolife.py
#
# Etiquettes de mode de transport de GeoLife (fichier Data/<user>/labels.txt, ~73 utilisateurs sur 182).
# Format : en-tete "Start Time<TAB>End Time<TAB>Transportation Mode", puis "2007/06/26 11:32:29<TAB>2007/06/26 11:40:29<TAB>bus".
#
# A VERIFIER AVANT D'INTERPRETER (hypotheses, non testees sur les vraies donnees) :
#   - fuseau : on suppose que les dates de labels.txt sont dans le meme referentiel UTC que les horodatages de sortie/
#     (decision 4 de DECISIONS.md : les .plt sont lus en UTC). `decalage_s` permet de corriger un decalage constant.
#   - regroupement en 5 classes (MAPPING_5) : a ALIGNER sur le tableau 3.3 du memoire (marche, velo, bus, voiture, train).
#     taxi -> voiture ; subway -> train ; les autres modes (airplane, boat, run, motorcycle) sont exclus.

import calendar
import os
import time

MAPPING_5 = {"walk": "marche", "bike": "velo", "bus": "bus", "car": "voiture", "taxi": "voiture",
             "train": "train", "subway": "train"}
CLASSES_5 = ["marche", "velo", "bus", "voiture", "train"]


def _epoch(texte):
    return calendar.timegm(time.strptime(texte.strip(), "%Y/%m/%d %H:%M:%S"))


def lire_labels(chemin, decalage_s=0.0):
    """Liste triee de (debut_epoch, fin_epoch, mode_brut). Fichier absent -> liste vide."""
    if not os.path.exists(chemin):
        return []
    sortie = []
    with open(chemin, encoding="utf-8") as f:
        for i, ligne in enumerate(f):
            champs = ligne.rstrip("\n").split("\t")
            if i == 0 or len(champs) < 3:
                continue
            try:
                sortie.append((_epoch(champs[0]) + decalage_s, _epoch(champs[1]) + decalage_s, champs[2].strip().lower()))
            except ValueError:
                continue
    return sorted(sortie)


def mode_du_segment(t_debut, t_fin, intervalles, seuil=0.8):
    """
    Mode brut qui couvre au moins `seuil` de la duree du segment, sinon None (segment mixte ou non etiquete).
    Les intervalles ne se chevauchent pas dans GeoLife ; en cas de chevauchement on additionne (borne par la duree).
    """
    duree = max(t_fin - t_debut, 1e-9)
    couverture = {}
    for a, b, m in intervalles:
        recouvrement = min(b, t_fin) - max(a, t_debut)
        if recouvrement > 0:
            couverture[m] = couverture.get(m, 0.0) + recouvrement
    if not couverture:
        return None
    m, c = max(couverture.items(), key=lambda kv: kv[1])
    return m if c / duree >= seuil else None
