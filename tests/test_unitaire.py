#!/usr/bin/env python3
# test_unitaire.py
"""
Script de test unitaire pour le prétraitement d'un seul fichier .plt.
Il applique l'intégralité du pipeline (nettoyage, segmentation, stats,
filtres, sauvegarde, logs, courbe) et écrit le résultat dans ./teste_pret/.

Utilisation :
    python test_unitaire.py --file chemin/vers/fichier.plt
ou bien en modifiant la variable FILE_PATH en bas du script.

Sortie :
    - teste_pret/<nom_fichier>/<nom_fichier>.csv
    - teste_pret/<nom_fichier>/log_segment_0.txt
    - teste_pret/<nom_fichier>/courbe_distance_0.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # racine du projet : permet `python tests/test_unitaire.py`

import os
import sys
import argparse
from datetime import datetime

# Import des modules du pipeline
from pretraitement.lecture import lire_fichier_plt
from pretraitement.nettoyage import nettoyer_points, supprimer_doublons
from pretraitement.tri import trier_par_temps
from pretraitement.segmentation import segmenter_points, filtrer_segments, filtrer_segments_stationnaires
from pretraitement.stats_segment import calculer_stats_segment, segment_en_dataframe
from pretraitement.filtres import filtrer_saut_distance, filtrer_vitesse
from pretraitement.sauvegarde import sauvegarder_utilisateur
from pretraitement.journal import ecrire_log_utilisateur
from pretraitement.courbe import tracer_courbes_utilisateur

from config import (
    TIME_GAP_THRESHOLD, MIN_POINTS, MAX_SPEED, DISTANCE_MIN,
    MAX_DIST_M, MODE_MAX_DIST, ORDRE_FILTRES,
    MAD_R_HETEROGENE, STATIO_RATIO_MIN, STATIO_RATIO_MAX
)

# ----------------------------------------------------------------------
# Fonctions locales (reprises de main.py mais simplifiées)
# ----------------------------------------------------------------------

def appliquer_filtres_intra_segment(df, stats):
    """
    Applique les filtres de Phase 3 (marquage) sur un segment.
    (copié depuis main.py pour rester cohérent)
    """
    if ORDRE_FILTRES == "distance_puis_vitesse":
        df = filtrer_saut_distance(df, MAX_DIST_M, mode=MODE_MAX_DIST)
        df = filtrer_vitesse(df, MAX_SPEED, stats_segment=stats)
    elif ORDRE_FILTRES == "vitesse_puis_distance":
        df = filtrer_vitesse(df, MAX_SPEED, stats_segment=stats)
        df = filtrer_saut_distance(df, MAX_DIST_M, mode=MODE_MAX_DIST)
    else:
        raise ValueError(f"config.ORDRE_FILTRES inconnu : {ORDRE_FILTRES!r}")
    return df


def traiter_un_fichier(file_path, output_dir="teste_pret"):
    """
    Traite un seul fichier .plt, applique tout le pipeline, et sauvegarde
    les résultats dans output_dir/<nom_base>/.
    """
    # 1. Extraire le nom de base du fichier (sans extension)
    basename = os.path.splitext(os.path.basename(file_path))[0]
    user_id = basename  # on utilise le nom du fichier comme "user_id" pour l'organisation

    # Dossier de sortie pour ce test
    out_user_dir = os.path.join(output_dir, user_id)
    os.makedirs(out_user_dir, exist_ok=True)

    print(f"\n--- Traitement du fichier : {file_path} ---")
    print(f"  Résultats dans : {out_user_dir}")

    # 2. Lecture du fichier
    points = lire_fichier_plt(file_path)
    if not points:
        print("  ERREUR : Aucun point lu.")
        return

    print(f"  Points lus : {len(points)}")

    # 3. Nettoyage, tri, suppression doublons
    points = nettoyer_points(points)
    points = trier_par_temps(points)
    points = supprimer_doublons(points)
    print(f"  Points après nettoyage/tri/dédoublonnage : {len(points)}")

    if len(points) < 2:
        print("  ERREUR : Trop peu de points après nettoyage.")
        return

    # 4. Segmentation temporelle
    segments_bruts = segmenter_points(points, TIME_GAP_THRESHOLD)
    print(f"  Segments bruts : {len(segments_bruts)}")

    # 5. Pour chaque segment : Phase 2, Phase 3, puis filtre final (étape 7)
    segments_valides = []          # pour le CSV final
    segments_info = []             # pour les logs

    for i, seg_points in enumerate(segments_bruts):
        # Phase 2 : statistiques
        stats = calculer_stats_segment(seg_points)
        df = segment_en_dataframe(seg_points)
        df.attrs["stats"] = stats   # pour les filtres relatifs éventuels

        # Phase 3 : marquage
        df = appliquer_filtres_intra_segment(df, stats)

        # Conserver le segment (même s'il sera peut-être filtré plus tard)
        segments_valides.append(df)
        segments_info.append({
            "df": df,
            "fichier_source": file_path,
            "stats": stats,
            "retenu": None  # sera mis à jour après le filtre final
        })

    # 6. Filtrage final (nombre de points et distance totale)
    segments_valides = filtrer_segments(segments_valides, MIN_POINTS)
    segments_valides = filtrer_segments_stationnaires(segments_valides, DISTANCE_MIN)

    # Mettre à jour le flag "retenu"
    ids_retenus = {id(df) for df in segments_valides}
    for info in segments_info:
        info["retenu"] = id(info["df"]) in ids_retenus

    print(f"  Segments après filtrage final : {len(segments_valides)}")

    for i, info in enumerate(segments_info):
        df = info["df"]
        n_art = int((df["type_suspect_vitesse"] == "artefact").sum())
        n_reg = int((df["type_suspect_vitesse"] == "regime").sum())
        alerte = bool(df["alerte_heterogene"].iloc[0])
        r_sta  = info["stats"].get("ratio_stationnarite")
        flag_sta = "*** NON STATIONNAIRE" if r_sta is not None and \
                not (STATIO_RATIO_MIN <= r_sta <= STATIO_RATIO_MAX) else "ok"
        print(f"  Seg {i} | retenu={info['retenu']} | "
            f"artefacts={n_art} regimes={n_reg} | "
            f"alerte_het={alerte} | statio={r_sta} {flag_sta}")

    # 7. Sauvegarde CSV (un seul fichier, comme pour un utilisateur)
    if segments_valides:
        sauvegarder_utilisateur(user_id, segments_valides, out_user_dir)

    # 8. Logs texte
    if segments_info:
        dossier_log = ecrire_log_utilisateur(user_id, segments_info, out_user_dir)
        print(f"  Logs écrits dans : {dossier_log}/")

        # 9. Courbes PNG
        chemins_png = tracer_courbes_utilisateur(user_id, segments_info, out_user_dir)
        print(f"  Courbes PNG : {len(chemins_png)} fichiers")

    print("--- Terminé ---\n")


# ----------------------------------------------------------------------
# Point d'entrée du script
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test unitaire du pipeline de prétraitement sur un seul fichier .plt."
    )
    parser.add_argument(
        "--file", type=str, required=True,
        help="Chemin vers le fichier .plt à traiter."
    )
    parser.add_argument(
        "--output", type=str, default="teste_pret",
        help="Dossier de sortie (défaut: teste_pret)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ERREUR : Fichier introuvable : {args.file}")
        sys.exit(1)

    traiter_un_fichier(args.file, args.output)


if __name__ == "__main__":
    main()