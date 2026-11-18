# ./pretraitement/journal.py
#
# NOUVEAU MODULE — journalisation détaillée du prétraitement, un
# fichier texte par segment.
#
# Inspiré du script de référence log_pretraitement_plt.py fourni par
# l'utilisateur, mais adapté à l'architecture en 3 phases du pipeline
# révisé :
#   - ce module n'effectue AUCUN filtrage : il ne fait qu'observer et
#     décrire, point par point, les marquages déjà produits par
#     pretraitement/filtres.py (colonnes "suspect_dist",
#     "suspect_vitesse"). Aucun point n'est jamais rejeté ni coupé ici ;
#   - l'export CSV de l'ancien script (exporter_csv_segments) n'est
#     PAS repris ici : les CSV produits par pretraitement/sauvegarde.py
#     restent l'unique sortie CSV du pipeline.
#
# Sortie produite par ce module :
#     log_segments/
#         <user_id>/
#             log_segment_0.txt
#             log_segment_1.txt
#             ...
#
# ============================================================
# V5 (cf. documentation/CONSTANTES.md) — modifications apportees :
# ------------------------------------------------------------
# 1. NETTOYAGE : la premiere definition de _construire_tableau_phases
#    (qui utilisait SEUIL_GAP_NOTABLE absolu) etait ECRASEE par la
#    seconde (qui utilise deja un seuil relatif 2.5 x pas median).
#    Bug silencieux, comportement non-documente. La premiere
#    definition est SUPPRIMEE ; seule la seconde est conservee.
#
# 2. SEUIL DE GAP : la constante SEUIL_GAP_NOTABLE (= 5 s absolu,
#    DEPRECATED dans config.py) est remplacee par le seuil RELATIF
#    max(SEUIL_GAP_NOTABLE_MIN, SEUIL_GAP_NOTABLE_RELATIF * pas_median).
#    Helper _seuil_gap_segment() centralise le calcul : la valeur
#    par defaut (2.5 x pas median) preserve la parite avec la V4.
#    Sur GeoLife (~7 s/pas median), le seuil effectif passe de 5 s
#    (V4, trop bas) a 17.5 s (V5, realiste).
#
# 3. SEUILS D'OBSERVATION : les seuils de vitesse hardcodes dans
#    _construire_tableau_phases (1.0, 8.0, 20.0, 50.0 km/h) sont
#    remplaces par JOURNAL_SEUILS_VITESSE_KMH et les labels par
#    JOURNAL_LABELS_VITESSE. Mêmes valeurs par defaut : aucun
#    changement de comportement, mais calibration centralisee.
#
# 4. IMPORTS : __import__("config").MAD_R_HETEROGENE est remplace
#    par un import propre en tete de module (deux occurrences).
#
# 5. EN-TETE DU LOG : ajout de trois lignes informatives pour la
#    relecture (pas median, statut de stationnarite KS, seuil de gap
#    effectif). Aucun impact algorithmique.
# ============================================================

import os
from datetime import datetime

import numpy as np

from config import (
    SEUIL_GAP_NOTABLE_MIN,
    SEUIL_GAP_NOTABLE_RELATIF,
    MAD_R_HETEROGENE,
    JOURNAL_SEUILS_VITESSE_KMH,
    JOURNAL_LABELS_VITESSE,
)
from pretraitement.geodesie import haversine_vectorise


def formater_timestamp(timestamp):
    """
    Convertit un timestamp UNIX (secondes) en (date_str, time_str)
    lisibles — l'inverse de utils.convertir_timestamp().

    Entrée :
        timestamp : float, secondes

    Sortie :
        (str, str) : ("AAAA-MM-JJ", "HH:MM:SS")
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")


def calculer_diagnostics_points(df):
    """
    Calcule, pour chaque point du segment, le delta_t, la distance et
    la vitesse instantanée par rapport au point PRÉCÉDENT DANS LE
    SEGMENT — qu'il soit marqué suspect ou non. C'est cohérent avec
    pretraitement/filtres.py, qui ne retire jamais un point de la
    séquence de référence (c'est précisément ce qui corrige le bogue
    de "comparaison vieillissante" de l'ancien pipeline).

    Implémentation vectorisée (numpy) : un seul calcul sur tout le
    segment, sur le même principe que stats_segment.py et filtres.py,
    plutôt qu'une boucle Python point par point. Ce calcul est
    redondant avec celui déjà fait dans filtres.py (qui ne conserve
    que les booléens suspect_*) : il est refait ici uniquement pour
    pouvoir afficher les valeurs numériques (distance, vitesse) dans
    le texte du log.

    Entrée :
        df : DataFrame du segment (colonnes "lat", "lon", "timestamp")

    Sortie :
        DataFrame : copie de df avec 3 colonnes ajoutées :
            "delta_t_s"    : intervalle avec le point précédent (s),
                             NaN pour le premier point
            "distance_m"   : distance avec le point précédent (m),
                             NaN pour le premier point
            "vitesse_inst" : vitesse instantanée (km/h), NaN si non
                             calculable (premier point, ou delta_t<=0)
    """
    df = df.copy()
    n = len(df)

    delta_t    = np.full(n, np.nan)
    distance_m = np.full(n, np.nan)
    vitesse    = np.full(n, np.nan)

    if n >= 2:
        lat = df["lat"].to_numpy()
        lon = df["lon"].to_numpy()
        t   = df["timestamp"].to_numpy()

        # np.diff(t) et haversine_vectorise() : même logique que dans
        # stats_segment.py / filtres.py (calcul en une passe sur tout
        # le segment plutôt qu'une boucle "for i in range(1, n)").
        dt_consec   = np.diff(t)
        dist_consec = haversine_vectorise(lat[:-1], lon[:-1], lat[1:], lon[1:]) * 1000.0

        delta_t[1:]    = dt_consec
        distance_m[1:] = dist_consec

        # dt<=0 -> vitesse indéfinie (NaN), comme dans filtres.py.
        dt_h = np.where(dt_consec > 0, dt_consec / 3600.0, np.nan)
        vitesse[1:] = (dist_consec / 1000.0) / dt_h

    df["delta_t_s"]    = delta_t
    df["distance_m"]   = distance_m
    df["vitesse_inst"] = vitesse

    return df


def _seuil_gap_segment(df_diag):
    """
    Seuil de gap effectif du segment : seuil RELATIF au pas median
    (V5), avec plancher absolu.

    Formule :
        seuil = max(SEUIL_GAP_NOTABLE_MIN,
                    SEUIL_GAP_NOTABLE_RELATIF * pas_median)

    Pourquoi relatif : un fichier GeoLife echantillonne en mode
    "temps" a un pas median ~3 s, en mode "distance" ~7 s. Un seuil
    absolu de 5 s (V4) signalait la moitie des intervalles normaux
    comme "gaps". Un seuil de 2.5 x pas median s'adapte au fichier.

    Cas limites :
        - segment sans dt valide (n < 2 ou dt tous NaN) ->
          retourne SEUIL_GAP_NOTABLE_MIN (plancher).
        - pas median = 0 (timestamps identiques, improbable apres
          supprimer_doublons) -> idem, plancher.

    Entree :
        df_diag : DataFrame du segment, sortie de
                  calculer_diagnostics_points() (colonne "delta_t_s").

    Sortie :
        float : seuil de gap en secondes.
    """
    dts = df_diag["delta_t_s"].dropna().to_numpy()
    if len(dts) == 0:
        return float(SEUIL_GAP_NOTABLE_MIN)
    pas_median = float(np.median(dts))
    if pas_median <= 0:
        return float(SEUIL_GAP_NOTABLE_MIN)
    return max(float(SEUIL_GAP_NOTABLE_MIN),
               float(SEUIL_GAP_NOTABLE_RELATIF) * pas_median)


def _categorie_vitesse(v_med):
    """
    Categorise une vitesse mediane (km/h) en label qualitatif.

    Utilise JOURNAL_SEUILS_VITESSE_KMH (bornes croissantes) et
    JOURNAL_LABELS_VITESSE (len(seuils)+1 labels). np.searchsorted
    realise la correspondance en une operation :
        v_med <  seuils[0]            -> labels[0]
        seuils[0] <= v_med < seuils[1] -> labels[1]
        ...
        v_med >= seuils[-1]           -> labels[-1]
    side="right" : v_med exactement egal a un seuil bascule vers la
    categorie SUPERIEURE (coherent avec la V4 : "v_med < 1.0" etait
    strict, donc v_med == 1.0 tombait dans "marche").

    Entree :
        v_med : float, vitesse mediane en km/h.

    Sortie :
        str : label qualitatif (parmi JOURNAL_LABELS_VITESSE).
    """
    bornes = np.asarray(JOURNAL_SEUILS_VITESSE_KMH, dtype=float)
    idx = int(np.searchsorted(bornes, v_med, side="right"))
    # Garde-fou si les listes ne sont pas alignees (ne devrait pas
    # arriver avec la config actuelle).
    idx = min(idx, len(JOURNAL_LABELS_VITESSE) - 1)
    return JOURNAL_LABELS_VITESSE[idx]


def _construire_tableau_phases(df_diag):
    """
    Construit le tableau de synthèse des phases d'un segment.

    Une "phase" est un bloc contigu de points dont le comportement est
    homogène. Un changement de phase est déclenché par l'un des trois
    événements suivants (cf. seuil_dyn.pdf §4.1) :
        a) Gap temporel notable : delta_t_s dépasse le seuil effectif
           max(SEUIL_GAP_NOTABLE_MIN, SEUIL_GAP_NOTABLE_RELATIF x
           pas_median) — seuil RELATIF au segment (V5). La V4 utilisait
           SEUIL_GAP_NOTABLE absolu (5 s), trop bas pour GeoLife.
        b) Début d'un régime soutenu : type_suspect_vitesse == "regime"
           (premier point d'un cluster de vitesse élevée soutenu,
           détecté par le filtre contextuel de filtrer_vitesse()).
        c) Saut de distance : suspect_dist == True.

    Pour chaque phase, on calcule :
        - Plage d'index (debut – fin)
        - Heure de début et de fin
        - Vitesse médiane et maximale (sur les points à dt valide)
        - Nombre de marquages (suspect_dist + suspect_vitesse)
        - Observation automatique (gap / régime / artefact / normal,
          plus catégorie qualitative de vitesse via
          JOURNAL_SEUILS_VITESSE_KMH).

    Entrée :
        df_diag : DataFrame, sortie de calculer_diagnostics_points(),
                  avec colonnes delta_t_s, vitesse_inst, suspect_dist,
                  suspect_vitesse, type_suspect_vitesse

    Sortie :
        list of str : lignes du tableau (format texte aligné)
    """
    n = len(df_diag)
    if n < 2:
        return ["  (segment trop court pour construire un tableau de phases)"]

    # Seuil de gap effectif du segment (V5 : relatif + plancher).
    seuil_gap = _seuil_gap_segment(df_diag)

    has_type = "type_suspect_vitesse" in df_diag.columns

    # ── Détection des coupures de phase ────────────────────────────
    # Une coupure à l'index i signifie que le point i commence une
    # nouvelle phase. On démarre toujours avec une coupure à i=0.
    coupures = [0]
    for i in range(1, n):
        est_gap    = (not np.isnan(df_diag["delta_t_s"].iloc[i])
                      and df_diag["delta_t_s"].iloc[i] > seuil_gap
                      and seuil_gap > 0)
        est_regime = (has_type
                      and df_diag["type_suspect_vitesse"].iloc[i] == "regime")
        est_saut   = bool(df_diag["suspect_dist"].iloc[i])

        if est_gap or est_regime or est_saut:
            coupures.append(i)

    coupures.append(n)  # sentinel de fin

    # ── Calcul des stats par phase ──────────────────────────────────
    phases = []
    for k in range(len(coupures) - 1):
        debut_idx = coupures[k]
        fin_idx   = coupures[k + 1] - 1
        bloc      = df_diag.iloc[debut_idx: fin_idx + 1]

        # Vitesses valides dans la phase (on exclut les NaN = 1er point)
        vits = bloc["vitesse_inst"].dropna().values
        vits = vits[~np.isnan(vits)]

        v_med = float(np.median(vits)) if len(vits) > 0 else 0.0
        v_max = float(np.max(vits))    if len(vits) > 0 else 0.0

        n_suspects = int(bloc["suspect_dist"].sum()
                         + bloc["suspect_vitesse"].sum())

        heure_debut = datetime.fromtimestamp(
            bloc["timestamp"].iloc[0]).strftime("%H:%M:%S")
        heure_fin   = datetime.fromtimestamp(
            bloc["timestamp"].iloc[-1]).strftime("%H:%M:%S")

        # Observation automatique : evenement structurel d'abord
        # (gap / regime / saut), puis categorie de vitesse (V5 :
        # centralisee via _categorie_vitesse).
        if has_type and df_diag["type_suspect_vitesse"].iloc[debut_idx] == "regime":
            obs = "debut_regime"
        elif (not np.isnan(df_diag["delta_t_s"].iloc[debut_idx])
              and df_diag["delta_t_s"].iloc[debut_idx] > seuil_gap
              and seuil_gap > 0):
            obs = "apres_gap"
        elif bool(df_diag["suspect_dist"].iloc[debut_idx]):
            obs = "apres_saut"
        else:
            obs = _categorie_vitesse(v_med)

        phases.append({
            "debut_idx": debut_idx,
            "fin_idx":   fin_idx,
            "n":         fin_idx - debut_idx + 1,
            "heure_debut": heure_debut,
            "heure_fin":   heure_fin,
            "v_med":     v_med,
            "v_max":     v_max,
            "n_suspects": n_suspects,
            "obs":       obs,
        })

    # ── Formatage du tableau ────────────────────────────────────────
    col_w = [10, 6, 9, 9, 8, 8, 8, 14]
    entetes = ["idx", "n", "debut", "fin", "v_med", "v_max",
               "suspects", "observation"]
    sep  = "  ".join("-" * w for w in col_w)
    head = "  ".join(f"{h:<{w}}" for h, w in zip(entetes, col_w))

    lignes = [f"  {head}", f"  {sep}"]
    for ph in phases:
        ligne = "  ".join([
            f"{ph['debut_idx']:>3}-{ph['fin_idx']:<4}"[:col_w[0]].ljust(col_w[0]),
            f"{ph['n']:<{col_w[1]}}",
            f"{ph['heure_debut']:<{col_w[2]}}",
            f"{ph['heure_fin']:<{col_w[3]}}",
            f"{ph['v_med']:>5.1f}  "[:col_w[4]+2].ljust(col_w[4]),
            f"{ph['v_max']:>5.1f}  "[:col_w[5]+2].ljust(col_w[5]),
            f"{ph['n_suspects']:<{col_w[6]}}",
            f"{ph['obs']:<{col_w[7]}}",
        ])
        lignes.append(f"  {ligne}")

    return lignes


def generer_lignes_log_segment(df_diag, seg_id, fichier_source, stats, retenu):
    """
    Construit le contenu texte du log d'un segment.

    Entrée :
        df_diag        : DataFrame du segment, sortie de
                         calculer_diagnostics_points() (colonnes
                         delta_t_s/distance_m/vitesse_inst en plus),
                         portant aussi "suspect_dist"/"suspect_vitesse"
                         (issues de pretraitement/filtres.py)
        seg_id         : int, identifiant du segment pour ce fichier
        fichier_source : str, chemin du fichier .plt d'origine
        stats          : dict, statistiques Phase 2
                         (stats_segment.calculer_stats_segment())
        retenu         : bool, True si ce segment a survécu au filtre
                         final (étape 7) et se trouve dans le CSV de
                         sortie (sortie/<user_id>.csv)

    Sortie :
        list of str : lignes du fichier de log
    """
    n = len(df_diag)
    nb_suspect_dist    = int(df_diag["suspect_dist"].sum())
    nb_suspect_vitesse = int(df_diag["suspect_vitesse"].sum())

    # Comptage par type (artefact / regime) si la colonne existe
    nb_artefact = nb_regime = 0
    if "type_suspect_vitesse" in df_diag.columns:
        nb_artefact = int((df_diag["type_suspect_vitesse"] == "artefact").sum())
        nb_regime   = int((df_diag["type_suspect_vitesse"] == "regime").sum())

    alerte_het = False
    if "alerte_heterogene" in df_diag.columns:
        alerte_het = bool(df_diag["alerte_heterogene"].iloc[0])

    # Seuil de gap effectif du segment (V5 : relatif, calcul centralise).
    seuil_gap = _seuil_gap_segment(df_diag)

    # delta_t_s > seuil_gap est faux pour les NaN (comparaison numpy
    # avec NaN -> False), donc le premier point n'est jamais compté
    # ici sans test explicite supplémentaire.
    nb_gaps_notables = int((df_diag["delta_t_s"] > seuil_gap).sum())

    lignes = []
    lignes.append(f"LOG DU SEGMENT {seg_id}")
    lignes.append(f"Fichier source : {fichier_source}")
    lignes.append(f"Segment retenu dans le CSV final : {'oui' if retenu else 'non'}")
    lignes.append(f"Nombre de points : {n}")
    lignes.append(f"Points marques suspect_dist    : {nb_suspect_dist}")
    lignes.append(f"Points marques suspect_vitesse : {nb_suspect_vitesse}")
    if nb_suspect_vitesse > 0:
        lignes.append(f"  dont artefacts GPS             : {nb_artefact}")
        lignes.append(f"  dont changements de regime     : {nb_regime}")
    if alerte_het:
        lignes.append(f"ALERTE heterogeneite structurelle : oui "
                      f"(sigma_v/v_med > {MAD_R_HETEROGENE})")

    # ── V5 : informations contextuelles (pas_median, statio KS) ─────
    # Ces lignes rendent le log AUTO-SUFFISANT pour relecture a froid :
    # on n'a plus besoin de recalculer le pas median ou de relancer le
    # test KS pour comprendre un segment 3 mois apres sa production.
    if stats.get("pas_median_s") is not None:
        lignes.append(f"Pas median (s)                : {stats['pas_median_s']:.2f}")
    if stats.get("statio_non_stationnaire") is not None:
        p = stats.get("statio_p_value")
        p_str = f" (p={p:.3f})" if p is not None else ""
        statut = "oui" if stats["statio_non_stationnaire"] else "non"
        lignes.append(f"Non-stationnaire (test KS)    : {statut}{p_str}")

    lignes.append(f"Seuil gap effectif (V5)       : {seuil_gap:.1f}s "
                  f"(= max({SEUIL_GAP_NOTABLE_MIN}s, "
                  f"{SEUIL_GAP_NOTABLE_RELATIF} x pas_median))")
    lignes.append(f"Gaps temporels notables (> {seuil_gap:.1f}s) : {nb_gaps_notables}")
    lignes.append("")
    lignes.append("--- Statistiques du segment (Phase 2) ---")

    if stats.get("mediane_vitesse") is not None:
        lignes.append(f"Vitesse mediane      : {stats['mediane_vitesse']:.1f} km/h")
        lignes.append(f"Ecart-type vitesse   : {stats['ecart_type_vitesse']:.1f} km/h")
        lignes.append(f"Vitesse max          : {stats['vitesse_max']:.1f} km/h")
        lignes.append(f"Distance mediane     : {stats['distance_mediane']*1000:.1f} m")
        lignes.append(f"Saut (distance max)  : {stats['distance_max']*1000:.1f} m")
        if stats.get("ratio_heterogeneite") is not None:
            r = stats["ratio_heterogeneite"]
            flag = " *** HETEROGENE" if r > MAD_R_HETEROGENE else ""
            lignes.append(f"Ratio sigma/mediane  : {r:.2f}{flag}")
        if stats.get("ratio_stationnarite") is not None:
            # V5 : le ratio est CONSERVE comme indicateur d'AFFICHAGE
            # (degrade : il ne pilote plus aucune decision). Le test
            # KS est la source de verite pour statio_non_stationnaire.
            rs = stats["ratio_stationnarite"]
            lignes.append(f"Ratio stationnarite  : {rs:.3f} (indicatif, cf. test KS)")
    else:
        lignes.append("Pas assez de points valides pour calculer ces statistiques")

    # ── Tableau de synthèse des phases ───────────────────────────────
    # Inséré ici, entre les statistiques globales et le détail point
    # par point (cf. seuil_dyn.pdf §4.1).
    #
    # Pourquoi ce tableau ?
    # Les statistiques globales (médiane, max, ratio R) résument tout
    # le segment en un seul chiffre. Le détail point par point est
    # exhaustif mais illisible pour comprendre la structure d'ensemble.
    # Le tableau des phases comble cet écart : il montre EN UN COUP
    # D'ŒIL combien de "blocs" comportementaux distincts existent
    # dans le segment, leur durée, leurs vitesses typiques, et les
    # marquages associés.
    lignes.append("")
    lignes.append("--- Synthese des phases ---")
    lignes += _construire_tableau_phases(df_diag)

    lignes.append("=" * 70)
    lignes.append("")

    for idx in range(n):
        ligne = df_diag.iloc[idx]
        date_str, time_str = formater_timestamp(ligne["timestamp"])

        v_inst_str = (f" | v_inst={ligne['vitesse_inst']:.1f}km/h"
                      if not np.isnan(ligne["vitesse_inst"]) else "")

        distance_str = (f" | distance={ligne['distance_m']:.1f}m"
                         if not np.isnan(ligne["distance_m"]) else "")

        delta_t_str = (f" | delta_t={ligne['delta_t_s']:.1f}s"
                        if not np.isnan(ligne["delta_t_s"]) else "")

        statuts = []
        if ligne["suspect_dist"]:
            statuts.append("suspect_dist")
        if ligne["suspect_vitesse"]:
            type_v = ""
            if "type_suspect_vitesse" in df_diag.columns:
                type_v = str(df_diag["type_suspect_vitesse"].iloc[idx])
            statuts.append(f"suspect_vitesse[{type_v}]" if type_v else "suspect_vitesse")
        statut_str = ",".join(statuts) if statuts else "ok"

        lignes.append(
            f"[{idx:5d}] {date_str} {time_str} | "
            f"lat={ligne['lat']:.6f} lon={ligne['lon']:.6f} alt={ligne['alt']:.0f} | "
            f"{statut_str}{distance_str}{delta_t_str}{v_inst_str}"
        )

        if idx == 0:
            lignes.append("          >> PREMIER POINT DU SEGMENT")

        # V5 : seuil_gap remplace SEUIL_GAP_NOTABLE (seuil relatif).
        if not np.isnan(ligne["delta_t_s"]) and ligne["delta_t_s"] > seuil_gap:
            lignes.append(
                f"          >> GAP TEMPOREL : delta_t={ligne['delta_t_s']:.1f}s "
                f"(seuil effectif={seuil_gap:.1f}s) distance={ligne['distance_m']:.1f}m"
            )

        if ligne["suspect_dist"]:
            lignes.append(
                f"          >> SUSPECT DISTANCE : saut de {ligne['distance_m']:.1f}m "
                f"detecte par filtrer_saut_distance() (point conserve, non coupe)"
            )

        if ligne["suspect_vitesse"]:
            type_v = ""
            if "type_suspect_vitesse" in df_diag.columns:
                type_v = str(df_diag["type_suspect_vitesse"].iloc[idx])
            label_type = f"[{type_v.upper()}]" if type_v else ""
            lignes.append(
                f"          >> SUSPECT VITESSE {label_type}: vitesse instantanee="
                f"{ligne['vitesse_inst']:.1f}km/h detectee par filtrer_vitesse() "
                f"(point conserve, non coupe)"
            )

    return lignes


def ecrire_log_utilisateur(user_id, segments_info, dossier_log_racine):
    """
    Écrit, pour un utilisateur, un fichier log_segment_X.txt par
    segment dans dossier_log_racine/user_id/.

    Entrée :
        user_id            : str, identifiant utilisateur (ex: "003")
        segments_info      : liste de dict, un par segment (dans
                              l'ordre où les segments ont été produits
                              par la Phase 1/3), avec les clés :
                                  "df"             : DataFrame marqué
                                                      (sortie Phase 3,
                                                      colonnes lat/lon/
                                                      alt/timestamp/
                                                      suspect_dist/
                                                      suspect_vitesse)
                                  "fichier_source" : str
                                  "stats"          : dict (Phase 2)
                                  "retenu"         : bool
        dossier_log_racine : str, ex. "log_segments"

    Sortie :
        str : chemin du dossier créé pour cet utilisateur
    """
    dossier_user = os.path.join(dossier_log_racine, user_id)
    os.makedirs(dossier_user, exist_ok=True)

    for seg_id, info in enumerate(segments_info):
        t = info["retenu"]

        df_diag = calculer_diagnostics_points(info["df"])
        lignes = generer_lignes_log_segment(
            df_diag, seg_id, info["fichier_source"], info["stats"], info["retenu"]
        )

        chemin = os.path.join(dossier_user, f"log_segment_{seg_id}.txt")
        with open(chemin, "w") as f:
            f.write("\n".join(lignes))

    return dossier_user