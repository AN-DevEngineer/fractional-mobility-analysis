# ./main.py
#
# Orchestration révisée du pipeline en TROIS PHASES distinctes,
# conformément à reflexion_revisee.pdf (§3) :
#
#   Phase 1 — nettoyage, tri, déduplication, segmentation temporelle
#   Phase 2 — statistiques par segment (calculer_stats_segment)
#   Phase 3 — filtrage intra-segment par MARQUAGE, jamais par
#             suppression (filtrer_saut_distance, filtrer_vitesse)
#
# Étape finale (7) — filtre sur les segments entiers : nombre de
# points (MIN_POINTS) et distance totale (DISTANCE_MIN, V5 : par
# segment via df.attrs["distance_min_km"]).
#
# Sortie : DEUX choses sont produites par utilisateur :
#   - sortie/<user_id>.csv             : les segments (marqués) ;
#   - log_segments/<user_id>/log_segment_X.txt : un log texte détaillé
#                                         par segment retenu.
#
# ============================================================
# V5 (cf. documentation/CONSTANTES.md) — modifications apportees :
# ------------------------------------------------------------
# 1. DISTANCE_MIN devient RELATIF au segment (Option B du plan V5).
#    Le seuil effectif de stationnarite est calcule PAR SEGMENT :
#       seuil_km = max(DISTANCE_MIN,
#                      MIN_POINTS * pas_median_s * V_PLANCHER / 3600)
#    et pose dans df.attrs["distance_min_km"] pour que
#    filtrer_segments_stationnaires() le lise sans changer sa
#    signature V4. Un segment de marche (pas median 5 s) et un
#    segment de bus (pas median 3 s) recoivent des seuils differents.
#    Impact GeoLife typique (7 s/pas) : seuil passe de 0.1 km a
#    0.583 km -> ~5.8x plus strict. Plus de segments filtres.
#    Si DISTANCE_MIN_RELATIF = False, comportement V4 conserve.
#
# 2. filtrer_saut_distance recoit maintenant max_dist_absolu_m =
#    MAX_DIST_M_ABSOLU (plafond absolu active en mode "relatif").
#    Le seuil effectif devient :
#       min(MAX_DIST_M * mediane_distance_m, MAX_DIST_M_ABSOLU)
#    Un saut de >500 m est donc TOUJOURS marque, meme sur un segment
#    a tres grand pas (TGV, avion).
#
# 3. Rapport console enrichi : distribution des seuils
#    DISTANCE_MIN effectifs (min / median / max) pour un utilisateur.
#    Permet de mesurer l'impact du changement sur le corpus.
#
# 4. BUG CORRIGE : la ligne sauvegarder_utilisateur() etait
#    indentee avec 8 espaces (a l'interieur du "if len(...)", elle
#    ne s'executait donc que si des segments existaient). Elle est
#    maintenant au niveau de la fonction (toujours executee).
# ============================================================

import os
from datetime import datetime

import numpy as np

from config import (
    TIME_GAP_THRESHOLD, MIN_POINTS, MAX_SPEED, DISTANCE_MIN,
    MAX_DIST_M, MODE_MAX_DIST, ORDRE_FILTRES,
    MAX_DIST_M_ABSOLU,
    DISTANCE_MIN_RELATIF, DISTANCE_MIN_V_PLANCHER_KMH,
)

from pretraitement.lecture       import lire_dossier_utilisateur
from pretraitement.nettoyage     import nettoyer_points, supprimer_doublons
from pretraitement.tri           import trier_par_temps
from pretraitement.segmentation  import segmenter_points, filtrer_segments, filtrer_segments_stationnaires
from pretraitement.stats_segment import calculer_stats_segment, segment_en_dataframe
from pretraitement.filtres       import filtrer_saut_distance, filtrer_vitesse
from pretraitement.sauvegarde    import sauvegarder_utilisateur
from pretraitement.journal       import ecrire_log_utilisateur
from pretraitement.courbe        import tracer_courbes_utilisateur


DATASET_PATH = os.path.join("dataset", "Geolife Trajectories 1.3", "Data")
SORTIE_PATH  = "sortie"
LOG_PATH     = "log_segments"

# V5 : arrêt propre du pipeline après ce délai (en secondes).
# Permet de libérer la machine / de laisser refroidir, tout en
# préservant la reprise (logique dernier_avec_logs ci-dessous).
TIMEOUT_SECONDS = 600  # 10 minutes

def format_duration(delta):
    """
    Convertit un objet timedelta en chaîne lisible du type :
        "1h 23m 45s" ou "45m 12s" ou "12s"
    """
    total_sec = int(delta.total_seconds())
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def _seuil_stationnaire_segment(stats):
    """
    Calcule le seuil DISTANCE_MIN EFFECTIF pour un segment (V5,
    Option B).

    Formule :
        seuil_km = max(DISTANCE_MIN,
                       MIN_POINTS * pas_median_s * V_PLANCHER / 3600)

    Justification : un segment de N points a pas median dt doit avoir
    parcouru au moins la distance qu'un "marcheur lente" (V_PLANCHER
    km/h) aurait parcourue dans le meme nombre de pas pour etre
    considere non-stationnaire. Un fichier a 5 s/pas et un fichier a
    60 s/pas n'ont pas le meme sens physique pour le meme "0.1 km".

    Si DISTANCE_MIN_RELATIF = False : retourne DISTANCE_MIN absolu
    (comportement V4).
    Si pas_median_s absent ou nul : fallback sur DISTANCE_MIN absolu.

    Entree :
        stats : dict, sortie de calculer_stats_segment()
                (cle "pas_median_s" utilisee).

    Sortie :
        float : seuil en km (>= DISTANCE_MIN).
    """
    if not DISTANCE_MIN_RELATIF:
        return float(DISTANCE_MIN)
    pas_median = stats.get("pas_median_s") if stats else None
    if not pas_median or pas_median <= 0:
        return float(DISTANCE_MIN)
    seuil_km = (MIN_POINTS * float(pas_median)
                * float(DISTANCE_MIN_V_PLANCHER_KMH) / 3600.0)
    return max(float(DISTANCE_MIN), seuil_km)


def appliquer_filtres_intra_segment(df, stats):
    """
    Applique les filtres de Phase 3 (marquage) sur un segment.

    L'ordre entre le filtre de distance (6a) et le filtre de vitesse
    (6b) n'est pas tranché par les documents de réflexion : les deux
    ordres doivent être expérimentés avant de choisir (reflexion_
    revisee.pdf, "Décisions arrêtées"). Le choix est donc piloté par
    config.ORDRE_FILTRES plutôt que figé en dur ici.

    V5 : filtrer_saut_distance recoit max_dist_absolu_m =
    MAX_DIST_M_ABSOLU, ce qui active le plafond absolu en mode
    "relatif" (voir filtres.py). Le seuil effectif devient
    min(MAX_DIST_M * mediane_distance_m, MAX_DIST_M_ABSOLU).

    Entrée :
        df    : DataFrame du segment (sortie de segment_en_dataframe())
        stats : dict, statistiques Phase 2 (calculer_stats_segment())

    Sortie :
        DataFrame : df avec les colonnes "suspect_dist" et
                    "suspect_vitesse" ajoutées
    """
    if ORDRE_FILTRES == "distance_puis_vitesse":
        df = filtrer_saut_distance(
            df, MAX_DIST_M, mode=MODE_MAX_DIST,
            max_dist_absolu_m=MAX_DIST_M_ABSOLU,
        )
        df = filtrer_vitesse(df, MAX_SPEED, stats_segment=stats)
    elif ORDRE_FILTRES == "vitesse_puis_distance":
        df = filtrer_vitesse(df, MAX_SPEED, stats_segment=stats)
        df = filtrer_saut_distance(
            df, MAX_DIST_M, mode=MODE_MAX_DIST,
            max_dist_absolu_m=MAX_DIST_M_ABSOLU,
        )
    else:
        raise ValueError(f"config.ORDRE_FILTRES inconnu : {ORDRE_FILTRES!r}")
    return df


def traiter_utilisateur(user_id, user_path):
    """
    Applique le pipeline complet (3 phases) sur un utilisateur,
    sauvegarde le CSV de sortie, et écrit un log détaillé par segment.

    Entrée :
        user_id   (str) : identifiant (ex: "003")
        user_path (str) : chemin vers le dossier de l'utilisateur

    Sortie :
        list : segments valides (DataFrames), marqués mais non coupés
               (ceux effectivement écrits dans sortie/<user_id>.csv)
    """
    print(f"\n=== Utilisateur {user_id} ===")

    tous_les_trajets = lire_dossier_utilisateur(user_path)
    print(f"  Fichiers lus : {len(tous_les_trajets)}")

    stats_globales = {
        "points_bruts":           0,
        "apres_nettoyage":        0,
        "apres_tri_doublons":     0,
        "segments_bruts":         0,
        "segments_valides":       0,
        "points_suspect_dist":      0,
        "points_suspect_vitesse":   0,
        "points_artefact":          0,
        "points_regime":            0,
        "segments_alerte_het":      0,
        # V5 : distribution des seuils DISTANCE_MIN effectifs
        # (pour mesurer l'impact du passage en relatif).
        "seuils_distance_min":      [],
    }

    tous_les_segments = []   # DataFrames marqués, pour le CSV de sortie
    segments_info      = []  # dicts (df, fichier_source, stats, retenu)

    for fichier_source, points in tous_les_trajets:

        stats_globales["points_bruts"] += len(points)

        # --- Phase 1 : nettoyage -> tri -> doublons -> segmentation ---
        points = nettoyer_points(points)
        stats_globales["apres_nettoyage"] += len(points)

        points = trier_par_temps(points)
        points = supprimer_doublons(points)
        stats_globales["apres_tri_doublons"] += len(points)

        if len(points) == 0:
            continue

        segments_bruts = segmenter_points(points, TIME_GAP_THRESHOLD)
        stats_globales["segments_bruts"] += len(segments_bruts)

        for segment in segments_bruts:
            # --- Phase 2 : statistiques du segment (numpy vectorisé) ---
            stats = calculer_stats_segment(segment)
            df = segment_en_dataframe(segment)
            df.attrs["stats"] = stats

            # --- V5 : seuil de stationnarite PAR SEGMENT (Option B) ---
            # Calcule une seule fois ici (le pas median est disponible
            # via stats["pas_median_s"]), pose dans df.attrs pour que
            # filtrer_segments_stationnaires() le lise en etape 7.
            # df.copy() (dans filtres.py) preserve les attrs, donc
            # l'attribut survit jusqu'au filtre final.
            seuil_km = _seuil_stationnaire_segment(stats)
            df.attrs["distance_min_km"] = seuil_km
            stats_globales["seuils_distance_min"].append(seuil_km)

            # --- Phase 3 : filtrage intra-segment par marquage ---
            df = appliquer_filtres_intra_segment(df, stats)

            stats_globales["points_suspect_dist"]    += int(df["suspect_dist"].sum())
            stats_globales["points_suspect_vitesse"] += int(df["suspect_vitesse"].sum())
            if "type_suspect_vitesse" in df.columns:
                stats_globales["points_artefact"] += int((df["type_suspect_vitesse"]=="artefact").sum())
                stats_globales["points_regime"]   += int((df["type_suspect_vitesse"]=="regime").sum())
            if "alerte_heterogene" in df.columns and bool(df["alerte_heterogene"].iloc[0]):
                stats_globales["segments_alerte_het"] += 1

            tous_les_segments.append(df)
            segments_info.append({
                "df":             df,
                "fichier_source": fichier_source,
                "stats":          stats,
                "retenu":         None,   # rempli après le filtre final
            })

    # --- Étape 7 : filtre final sur les segments entiers ---
    # Les deux critères (nombre de points et distance totale) sont
    # appliqués l'un après l'autre.
    tous_les_segments = filtrer_segments(tous_les_segments, MIN_POINTS)
    # V5 : chaque segment porte son propre seuil dans
    # attrs["distance_min_km"]. On passe DISTANCE_MIN en fallback
    # (sécurité, ne devrait jamais être lu). Si un segment n'a pas
    # l'attribut, distance_min_km global s'applique.
    tous_les_segments = filtrer_segments_stationnaires(tous_les_segments, DISTANCE_MIN)
    stats_globales["segments_valides"] = len(tous_les_segments)

    # filtrer_segments() / filtrer_segments_stationnaires() ne copient
    # jamais les DataFrames : elles ne font que sélectionner un
    # sous-ensemble de la liste reçue. id(df) permet donc de retrouver,
    # par identité d'objet, quels segments de segments_info ont
    # survécu au filtre final.
    ids_retenus = {id(df) for df in tous_les_segments}
    for info in segments_info:
        info["retenu"] = id(info["df"]) in ids_retenus

    # Rapport de qualité
    bruts = stats_globales["points_bruts"]
    if bruts > 0:
        print(f"  Points bruts            : {bruts:>8,}")
        print(f"  Après nettoyage         : {stats_globales['apres_nettoyage']:>8,} "
              f"({100*stats_globales['apres_nettoyage']/bruts:.1f}%)")
        print(f"  Après tri/doublons      : {stats_globales['apres_tri_doublons']:>8,} "
              f"({100*stats_globales['apres_tri_doublons']/bruts:.1f}%)")
        print(f"  Segments bruts          : {stats_globales['segments_bruts']:>8,}")
        print(f"  Segments valides        : {stats_globales['segments_valides']:>8,}")

        # V5 : distribution des seuils DISTANCE_MIN effectifs.
        seuils = stats_globales["seuils_distance_min"]
        if seuils:
            print(f"  Seuils DISTANCE_MIN effectifs (km) : "
                  f"min={min(seuils):.3f}  "
                  f"méd={float(np.median(seuils)):.3f}  "
                  f"max={max(seuils):.3f}")

        # Segments écartés : comptés (pas journalisés) pour choisir
        # MIN_POINTS et DISTANCE_MIN en connaissance de cause.
        segments_ecartes = [info for info in segments_info if not info["retenu"]]
        nb_trop_courts = sum(1 for info in segments_ecartes if len(info["df"]) < MIN_POINTS)
        # V5 : le libellé "distance < {DISTANCE_MIN}" est remplacé par
        # "distance < seuil effectif" : le seuil varie d'un segment à
        # l'autre, on affiche juste le nombre de segments écartés par
        # ce critère (ceux qui ont passé le filtre de longueur).
        print(f"  Segments écartés        : {len(segments_ecartes):>8,} "
              f"(< {MIN_POINTS} points : {nb_trop_courts:,} ; "
              f"distance < seuil effectif : {len(segments_ecartes) - nb_trop_courts:,})")
        print(f"  Points marqués suspect_dist            : {stats_globales['points_suspect_dist']:>8,}")
        print(f"  Points marqués suspect_vitesse         : {stats_globales['points_suspect_vitesse']:>8,}")
        print(f"    dont artefacts GPS                   : {stats_globales['points_artefact']:>8,}")
        print(f"    dont changements de regime           : {stats_globales['points_regime']:>8,}")
        print(f"  Segments alerte heterogeneite          : {stats_globales['segments_alerte_het']:>8,}")

    # Logs et courbes : UNIQUEMENT pour les segments retenus.
    segments_retenus = [info for info in segments_info if info["retenu"]]

    # Le dossier de logs marque l'utilisateur comme "déjà traité".
    # On le crée toujours, même si aucun segment n'est retenu.
    os.makedirs(os.path.join(LOG_PATH, user_id), exist_ok=True)

    if len(segments_retenus) > 0:
        dossier_log = ecrire_log_utilisateur(user_id, segments_retenus, LOG_PATH)
        print(f"  Logs écrits : {dossier_log}/ ({len(segments_retenus)} fichier(s))")
        chemins_png = tracer_courbes_utilisateur(user_id, segments_retenus, LOG_PATH)
        print(f"  Courbes     : {len(chemins_png)} image(s) PNG")

    # V5 : BUG CORRIGE. La ligne était indentée à 8 espaces (dans le
    # "if len(...)"), donc l'export CSV ne se faisait que si des
    # segments existaient. Elle est maintenant au niveau de la
    # fonction : toujours exécutée, même si le fichier CSV produit
    # est vide (ce qui est cohérent avec la logique de reprise de
    # main() qui teste os.path.getsize(output_csv) > 0).
    sauvegarder_utilisateur(user_id, tous_les_segments, SORTIE_PATH)

    return tous_les_segments


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"[ERREUR] Dataset introuvable : {DATASET_PATH}")
        return

    os.makedirs(SORTIE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    # --- Temps de début ---
    start_time = datetime.now()
    print(f"🚀 Début du pipeline : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Temps limite : {TIMEOUT_SECONDS} secondes ({TIMEOUT_SECONDS//60} minutes)")
    print("-" * 60)

    utilisateurs = sorted(os.listdir(DATASET_PATH))
    print(f"Utilisateurs trouvés : {len(utilisateurs)}")

    # --- Identifier le dernier utilisateur avec logs ---
    dernier_avec_logs = None
    for nom in utilisateurs:
        log_dir = os.path.join(LOG_PATH, nom)
        if os.path.exists(log_dir):
            dernier_avec_logs = nom

    if dernier_avec_logs is not None:
        print(f"Dernier utilisateur avec logs : {dernier_avec_logs}")

    total_segments = 0
    traites = 0
    ignores = 0
    retraites = 0
    timeout_atteint = False

    try:
        for idx, nom_user in enumerate(utilisateurs):
            user_path = os.path.join(DATASET_PATH, nom_user)

            if not os.path.isdir(user_path):
                continue

            # --- V5 : Vérification du timeout AVANT de commencer un nouvel utilisateur ---
            # Placé APRÈS le check isdir, pour ne pas "consommer" le timeout
            # sur un dossier parasite (ex: .ipynb_checkpoints, etc.).
            # Placé AVANT le check "logs présents", pour que le timeout
            # puisse couper même si l'utilisateur courant était ignoré.
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > TIMEOUT_SECONDS:
                print(f"\n⏱️  Temps limite atteint ({elapsed:.0f}s > {TIMEOUT_SECONDS}s).")
                print("Arrêt programmé. Les données déjà traitées sont conservées.")
                print("Relancez le script pour reprendre à l'utilisateur suivant.")
                timeout_atteint = True
                break

            log_dir = os.path.join(LOG_PATH, nom_user)
            output_csv = os.path.join(SORTIE_PATH, f"{nom_user}.csv")

            # --- Cas 1 : pas de logs → jamais traité ---
            if not os.path.exists(log_dir):
                print(f"\n--- Traitement de l'utilisateur {nom_user} (première fois) ---")
                user_start = datetime.now()
                segments = traiter_utilisateur(nom_user, user_path)
                user_end = datetime.now()
                user_dur = format_duration(user_end - user_start)
                print(f"  ✅ Temps pour {nom_user} : {user_dur}")
                total_segments += len(segments)
                traites += 1
                continue

            # --- Cas 2 : logs présents ---
            # Si ce n'est pas le dernier avec logs, on ignore
            if nom_user != dernier_avec_logs:
                print(f"  Utilisateur {nom_user} déjà traité (logs présents), ignoré.")
                ignores += 1
                continue

            # --- Cas 3 : c'est le dernier avec logs → vérifier le CSV ---
            if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
                print(f"  Utilisateur {nom_user} (dernier logs) : CSV présent, ignoré.")
                ignores += 1
            else:
                print(f"  ⚠ Utilisateur {nom_user} (dernier logs) : CSV manquant/vide → retraitement.")
                print(f"\n--- Retraitement de l'utilisateur {nom_user} ---")
                user_start = datetime.now()
                segments = traiter_utilisateur(nom_user, user_path)
                user_end = datetime.now()
                user_dur = format_duration(user_end - user_start)
                print(f"  ✅ Temps pour {nom_user} : {user_dur}")
                total_segments += len(segments)
                retraites += 1

    except KeyboardInterrupt:
        elapsed = format_duration(datetime.now() - start_time)
        print("\n" + "=" * 60)
        print("⏹️  Interruption clavier détectée (Ctrl+C).")
        print("Arrêt propre du pipeline en cours...")
        print(f"  Utilisateurs traités   : {traites} (première fois)")
        print(f"  Utilisateurs retraités : {retraites} (dernier logs, CSV manquant)")
        print(f"  Utilisateurs ignorés   : {ignores} (logs présents, sauf dernier avec CSV)")
        print(f"  Total segments sauvegardés : {total_segments}")
        print(f"  ⏱️  Temps total écoulé : {elapsed}")
        print("=" * 60)
        print("✅ Données déjà sauvegardées. Relancez le script pour continuer.")
        return

    # --- Fin normale (ou timeout) ---
    elapsed = format_duration(datetime.now() - start_time)
    if timeout_atteint:
        print(f"\n=== Pipeline interrompu par TIMEOUT ({TIMEOUT_SECONDS}s) ===")
    else:
        print(f"\n=== Pipeline terminé ===")
    print(f"  Utilisateurs traités   : {traites} (première fois)")
    print(f"  Utilisateurs retraités : {retraites} (dernier logs, CSV manquant)")
    print(f"  Utilisateurs ignorés   : {ignores} (logs présents, sauf dernier avec CSV)")
    print(f"  Total segments sauvegardés : {total_segments}")
    print(f"  Fichiers CSV dans      : {SORTIE_PATH}/")
    print(f"  Logs par segment dans  : {LOG_PATH}/")
    print(f"  ⏱️  Temps total écoulé : {elapsed}")
    if timeout_atteint:
        print("  ℹ️  Relancez le script pour reprendre là où vous en étiez.")

if __name__ == "__main__":
    main()