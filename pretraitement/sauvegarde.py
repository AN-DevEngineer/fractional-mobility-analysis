# ./pretraitement/sauvegarde.py
#
# Réécrit avec pandas pour persister les colonnes de marquage
# ("suspect_dist", "suspect_vitesse") ajoutées en Phase 3. Ce
# marquage doit rester visible dans le CSV de sortie : c'est ce qui
# permet d'observer l'impact des points suspects sur une estimation de
# H avant de décider d'une éventuelle coupure (question encore ouverte
# dans les documents de réflexion).

import os

import pandas as pd


def sauvegarder_utilisateur(user_id, segments, dossier_sortie):
    """
    Sauvegarde tous les segments d'un utilisateur dans un fichier CSV

    Structure du fichier : un CSV par utilisateur
    Colonnes : segment_id, point_id, lat, lon, alt, timestamp,
               suspect_dist, suspect_vitesse
    (les deux dernières colonnes sont absentes si les segments fournis
    n'ont pas encore été marqués par pretraitement/filtres.py)

    Entrée :
        user_id        (str) : identifiant de l'utilisateur (ex: "003")
        segments       (list): liste de DataFrames (un par segment),
                               colonnes "lat", "lon", "alt", "timestamp"
                               et optionnellement "suspect_dist",
                               "suspect_vitesse"
        dossier_sortie (str) : chemin vers le dossier de sortie

    Sortie :
        str : chemin du fichier créé
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    chemin_fichier = os.path.join(dossier_sortie, f"{user_id}.csv")
    
    # V5 : cas segments vide (aucun segment n'a passé le filtre final).
    # On écrit un CSV avec les EN-TÊTES SEULS (0 ligne). Pourquoi :
    #   - évite le crash pd.concat([]) ;
    #   - le fichier existe avec getsize > 0, donc la logique de
    #     reprise (test getsize(output_csv) > 0) ne retraitera pas ce
    #     user indéfiniment ;
    #   - 03_estimation_H.py saura lire ce fichier et skipper l'user
    #     proprement (0 segment).
    if len(segments) == 0:
        colonnes = ["segment_id", "point_id", "lat", "lon", "alt",
                    "timestamp", "suspect_dist", "suspect_vitesse",
                    "type_suspect_vitesse", "alerte_heterogene"]
        pd.DataFrame(columns=colonnes).to_csv(chemin_fichier, index=False)
        print(f"  Sauvegardé : {chemin_fichier} "
              f"(0 segment retenu — en-têtes seuls)")
        return chemin_fichier

    morceaux = []
    for segment_id, df in enumerate(segments):
        # df.copy() : on ne modifie jamais le DataFrame reçu en entrée
        # (effet de bord évité), puis df.insert() ajoute les colonnes
        # d'identification segment_id/point_id en tête de colonnes.
        df = df.copy()
        df.insert(0, "point_id", range(len(df)))
        df.insert(0, "segment_id", segment_id)
        morceaux.append(df)

    # pd.concat() rassemble tous les segments en un seul DataFrame
    # avant l'écriture : une seule opération d'I/O vectorisée
    # (to_csv) remplace la double boucle "for segment ... for point ..."
    # avec un writer.writerow() par ligne de l'ancienne implémentation.
    table = pd.concat(morceaux, ignore_index=True)

    # Ordre de colonnes fixe pour la lisibilité du CSV ; les colonnes
    # de marquage ne sont incluses que si elles existent (segments non
    # encore passés par filtres.py, par exemple en cas d'usage hors du
    # pipeline complet).
    colonnes_attendues = ["segment_id", "point_id", "lat", "lon", "alt",
                          "timestamp", "suspect_dist", "suspect_vitesse",
                          "type_suspect_vitesse", "alerte_heterogene"]
    colonnes = [c for c in colonnes_attendues if c in table.columns]
    table = table[colonnes]

    table.to_csv(chemin_fichier, index=False)

    print(f"  Sauvegardé : {chemin_fichier} ({len(segments)} segments)")
    return chemin_fichier


def charger_utilisateur(user_id, dossier_sortie):
    """
    Charge le CSV d'un utilisateur et reconstruit la liste de segments

    Entrée :
        user_id        (str) : identifiant de l'utilisateur (ex: "003")
        dossier_sortie (str) : chemin vers le dossier de sortie

    Sortie :
        list : liste de DataFrames (un par segment), colonnes "lat",
               "lon", "alt", "timestamp" et, si présentes dans le CSV,
               "suspect_dist", "suspect_vitesse"
        Retourne [] si le fichier n'existe pas
    """
    chemin_fichier = os.path.join(dossier_sortie, f"{user_id}.csv")

    if not os.path.exists(chemin_fichier):
        print(f"[ERREUR] Fichier introuvable : {chemin_fichier}")
        return []

    table = pd.read_csv(chemin_fichier)

    segments = []
    # table.groupby("segment_id", sort=False) regroupe toutes les
    # lignes d'un même segment en un seul DataFrame par groupe — cela
    # remplace la boucle manuelle de l'ancienne implémentation qui
    # détectait elle-même les changements de segment_id ligne par
    # ligne. sort=False conserve l'ordre d'écriture plutôt que de
    # retrier les segments par id.
    for _, groupe in table.groupby("segment_id", sort=False):
        df = groupe.drop(columns=["segment_id", "point_id"]).reset_index(drop=True)
        segments.append(df)

    return segments
