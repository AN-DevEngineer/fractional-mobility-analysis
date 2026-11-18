# ./pretraitement/filtres.py
#
# Phase 3 du pipeline révisé : filtrage intra-segment par MARQUAGE.
#
# Deux corrections apportées par rapport à la version précédente
# (cf. seuil_dyn.pdf, version du 6 juillet 2026) :
#
# CORRECTION 1 — Asymétrie du Z-score (§2.2 de seuil_dyn.pdf)
#   L'ancienne version calculait |Z_i| sur toutes les vitesses, ce qui
#   pouvait marquer des vitesses anormalement BASSES (ralentissements,
#   arrêts) comme suspectes. Or une vitesse de 0,2 km/h dans un trajet
#   de marche n'est pas un artefact — c'est un arrêt normal.
#   Correction : le Z-score n'est calculé que pour v_i > ṽ. Les vitesses
#   inférieures ou égales à la médiane ne sont JAMAIS candidates.
#
# CORRECTION 2 — Marquage du premier point du cluster uniquement (§2.3)
#   L'ancienne version marquait TOUS les points d'un cluster "regime"
#   (ex. 125 points de bus = 125 lignes suspect_vitesse=True dans le CSV
#   et le log), ce qui polluait les sorties sans apporter d'information
#   supplémentaire. Correction : seul le PREMIER point du cluster est
#   marqué avec type_suspect_vitesse="regime". Les suivants restent ok.
#   Un attribut global alerte_heterogene=True signale le segment entier.
#
# ============================================================
# V5 (cf. documentation/CONSTANTES.md) — seuils RELATIFS
# ------------------------------------------------------------
# Toutes les decisions de marquage impliquent maintenant un seuil
# qui PEUT ETRE RELATIF au contexte du segment (mediane, MAD, N).
# Les valeurs absolues de config.py sont conservees comme
# PLANCHERS ou PLAFONDS (garde-fous). Les nouvelles constantes sont
# lues depuis config.py a l'import.
#
# 1. filtrer_saut_distance : en mode "relatif",
#       seuil = min(MAX_DIST_M * mediane_distance_m, MAX_DIST_M_ABSOLU)
#    Le min() garantit qu'on n'excede JAMAIS le plafond absolu, meme
#    sur un segment a tres grand pas (ex. TGV). Le parametre
#    max_dist_absolu_m est nouveau ; s'il est None, comportement V4
#    est conserve (seuil = MAX_DIST_M * mediane, sans plafond).
#
# 2. filtrer_vitesse : quatre seuils rendus relatifs.
#       seuil_v_max     = min(MAX_SPEED, max(MAD_FALLBACK_SPEED,
#                                            MAX_SPEED_RELATIF * v_med))
#                          (min() : la valeur la plus stricte l'emporte.
#                           Corrige la formule erronee "max(...)" du
#                           plan V5 : max(200, 20*3)=200 NE protege PAS
#                           un segment de marche ou 60 km/h est deja
#                           aberrant ; min(200,60)=60 est correct.)
#                          (plancher MAD_FALLBACK_SPEED : evite un seuil
#                           absurde sur segments quasi-stationnaires.)
#       seuil_persist   = max(MAD_PERSISTANCE_MIN,
#                             round(MAD_PERSISTANCE_FRACTION * n))
#       seuil_dv        = max(MAD_DELTA_V_BRUTAL,
#                             MAD_DELTA_V_FACTEUR_MAD * mad)
#       seuil_fallback  = max(MAD_FALLBACK_SPEED,
#                             MAD_FALLBACK_SPEED_FACTEUR * v_med)
#    Les max() sont des planchers (ne pas etre TROP sensible sur un
#    segment tres regulier) ; voir commentaires internes.
#
# 3. Le Z-score asymetrique (MAD_Z_SEUIL = 3.5) reste ABSOLU :
#    il est soutenu par la litterature (Iglewicz & Hoaglin 1993) et
#    son facteur 0.6745 est mathematique, pas empirique. Le MAD est
#    DEJA une mesure robuste et relative au segment : la formule
#    Z = 0.6745*(v - v_med)/MAD s'adapte deja a la dispersion locale.
#    Un facteur contextuel supplementaire rendrait le detecteur
#    incoherent (le MAD est precisement ce qui donne l'echelle).
#
# 4. Import de MAD_R_HETEROGENE en haut de fichier, suppression du
#    __import__("config") inline (bidouille de la V4).
#
# 5. alerte_heterogene : la constante MAD_R_TEST_NUL existe dans
#    config.py mais son implementation (comparaison au quantile 95%
#    d'une distribution nulle via validation/modeles_nuls.py) est
#    prevue dans une etape ulterieure. En attendant, on conserve le
#    seuil absolu MAD_R_HETEROGENE (comportement V4).
# ============================================================

import numpy as np

from config import (
    MAD_Z_SEUIL, MAD_PERSISTANCE_MIN, MAD_PERSISTANCE_FRACTION,
    MAD_DELTA_V_BRUTAL, MAD_DELTA_V_FACTEUR_MAD,
    MAD_R_HETEROGENE, MAD_R_TEST_NUL,
    MAD_FALLBACK_SPEED, MAD_FALLBACK_SPEED_FACTEUR,
    MAX_SPEED_RELATIF,
)
from pretraitement.geodesie import haversine_vectorise


def filtrer_saut_distance(df, max_dist_m, mode="absolu",
                          max_dist_absolu_m=None):
    """
    Marque (filtre 6a) les points dont la distance au point précédent
    dépasse un seuil.

    mode="absolu"  : max_dist_m est une distance fixe en mètres.
    mode="relatif" : max_dist_m est un multiplicateur de la distance
                     médiane du segment (stats Phase 2 requises via
                     df.attrs["stats"], clé "mediane_distance_m").

    V5 : en mode "relatif", le seuil effectif est
        min(max_dist_m * mediane_distance_m, max_dist_absolu_m)
    Le min() applique la valeur la plus stricte des deux. Le plafond
    absolu (max_dist_absolu_m) est un garde-fou : il empêche un segment
    à très grand pas (ex. TGV, 10 km/pas) de laisser passer un saut de
    5 km sous prétexte que 30 × 10 km = 300 km. Si max_dist_absolu_m
    est None, la V4 est conservée (pas de plafond).

    Entrée :
        df                 : DataFrame du segment (colonnes lat, lon,
                             alt, timestamp ; df.attrs["stats"] peuplé
                             par calculer_stats_segment() en Phase 2)
        max_dist_m         : float, seuil absolu (mode "absolu") ou
                             multiplicateur (mode "relatif")
        mode               : "absolu" | "relatif"
        max_dist_absolu_m  : float ou None, plafond absolu (V5, mode
                             "relatif" seulement)

    Sortie :
        DataFrame avec colonne "suspect_dist" (bool) ajoutée.
        Toujours False sur le premier point (pas de prédécesseur).
    """
    n = len(df)
    suspect = np.zeros(n, dtype=bool)

    if n >= 2:
        lat = df["lat"].to_numpy()
        lon = df["lon"].to_numpy()

        # Distances vectorisées entre chaque point et son prédécesseur
        # immédiat — une seule passe numpy remplace la boucle Python.
        distances_m = haversine_vectorise(lat[:-1], lon[:-1],
                                          lat[1:], lon[1:]) * 1000.0

        if mode == "absolu":
            seuil_m = max_dist_m

        elif mode == "relatif":
            stats = df.attrs.get("stats") or {}
            # V5 : on lit "mediane_distance_m" (source de verite, deja
            # convertie en metres par calculer_stats_segment).
            # Fallback sur "distance_mediane" (V4, en km) pour
            # compatibilite si un appelant n'a pas la nouvelle cle.
            mediane_m = stats.get("mediane_distance_m")
            if mediane_m is None:
                mediane_km = stats.get("distance_mediane")
                if mediane_km:
                    mediane_m = mediane_km * 1000.0

            if not mediane_m or mediane_m <= 0:
                # Reference indisponible ou segment immobile (toutes
                # distances = 0) :
                #   - si plafond absolu fourni -> on l'utilise (garde-fou)
                #   - sinon -> on ne marque rien (comportement V4)
                seuil_m = (max_dist_absolu_m
                           if max_dist_absolu_m is not None
                           else np.inf)
            else:
                seuil_relatif = max_dist_m * mediane_m
                if max_dist_absolu_m is not None:
                    # V5 : min() = on applique la valeur la plus stricte.
                    # Le plafond empeche un segment a tres grand pas de
                    # laisser passer un saut enorme "au prorata".
                    seuil_m = min(seuil_relatif, max_dist_absolu_m)
                else:
                    seuil_m = seuil_relatif

        else:
            raise ValueError(
                f"mode inconnu pour filtrer_saut_distance : {mode!r}")

        # Comparaison vectorisée en une seule passe.
        suspect[1:] = distances_m > seuil_m

    df = df.copy()
    df["suspect_dist"] = suspect
    return df


def filtrer_vitesse(df, max_speed_kmh, stats_segment=None):
    """
    Marque (filtre 6b) les points dont la vitesse instantanée est
    aberrante. Approche en trois niveaux.

    ─────────────────────────────────────────────────────────────────
    NIVEAU 1 — Détecteur MAD (Z-score modifié ASYMÉTRIQUE)
    ─────────────────────────────────────────────────────────────────
    On calcule la médiane ṽ et le MAD des vitesses du segment :

        ṽ   = médiane(v_i)
        MAD = médiane(|v_i − ṽ|)
        Z_i = 0.6745 × (v_i − ṽ) / MAD

    Le facteur 0.6745 (= Φ^{-1}(3/4)) ramène dans l'échelle d'un
    z-score standard pour une distribution normale (Iglewicz & Hoaglin,
    1993 ; Leys et al., 2013).

    ASYMÉTRIE (correction seuil_dyn.pdf §2.2) : Z_i est calculé
    UNIQUEMENT pour v_i > ṽ. Les vitesses inférieures ou égales à la
    médiane ne peuvent jamais être candidates.

    Un point est candidat si Z_i > MAD_Z_SEUIL (3.5 par défaut) OU
    si v_i > seuil_v_max (garde-fou absolu ET contextuel, voir V5).

    ─────────────────────────────────────────────────────────────────
    NIVEAU 2 — Filtre contextuel (variation locale + persistance)
    ─────────────────────────────────────────────────────────────────
    Pour chaque candidat, deux critères sont examinés :

    a) Variation locale Δv = |v_i − v_{i−1}| :
       Si Δv > seuil_dv (V5 : max plancher absolu et k×MAD_v), le pic
       est "brutal" → artefact GPS. Sinon, "progressif" → possible
       changement de mode.

    b) Persistance : longueur de la séquence de candidats consécutifs.
       Si ≥ seuil_persistance (V5 : max plancher et fraction × N) →
       régime soutenu (bus, voiture). Sinon → bruit local, non marqué.

    MARQUAGE DU PREMIER POINT UNIQUEMENT pour les régimes (correction
    seuil_dyn.pdf §2.3).

    Trois catégories résultantes (colonne type_suspect_vitesse) :
        "artefact" : pic brutal et isolé      → suspect_vitesse=True
        "regime"   : début d'un régime soutenu → suspect_vitesse=True
        ""         : bruit local ou v ≤ ṽ     → suspect_vitesse=False

    ─────────────────────────────────────────────────────────────────
    NIVEAU 3 — Alerte globale d'hétérogénéité (ratio R)
    ─────────────────────────────────────────────────────────────────
    Si ratio_heterogeneite = σ_v / ṽ > MAD_R_HETEROGENE (1.5), la
    colonne alerte_heterogene est mise à True sur tout le segment.
    Aucun point individuel n'est marqué supplémentairement, mais le log
    signale que l'estimation de H doit être interprétée avec précaution
    (cf. seuil_dyn.pdf §2.4).

    NOTE V5 : MAD_R_TEST_NUL existe dans config.py mais son
    implementation (quantile 95% de la distribution nulle) est prevue
    dans une etape ulterieure. En attendant, on conserve le seuil
    absolu MAD_R_HETEROGENE.

    ─────────────────────────────────────────────────────────────────
    GARDE-FOUS
    ─────────────────────────────────────────────────────────────────
    - MAD = 0 (segment quasi-immobile) → fallback absolu
      seuil_fallback (V5 : max plancher absolu et k × v_med) pour
      éviter une division par zéro.
    - max_speed_kmh conservé comme seuil absolu de dernier recours
      (via seuil_v_max) : tout point le dépassant est "artefact".

    Entrée :
        df            : DataFrame du segment
        max_speed_kmh : float, seuil absolu de dernier recours (km/h)
        stats_segment : dict ou None (statistiques Phase 2, pour R)

    Sortie :
        DataFrame avec trois colonnes ajoutées :
            suspect_vitesse      : bool
            type_suspect_vitesse : str, "artefact" | "regime" | ""
            alerte_heterogene    : bool (identique sur tout le segment)
    """
    n = len(df)
    suspect      = np.zeros(n, dtype=bool)
    type_suspect = np.full(n, "", dtype=object)

    if n >= 2:
        lat = df["lat"].to_numpy()
        lon = df["lon"].to_numpy()
        t   = df["timestamp"].to_numpy()

        delta_t   = np.diff(t)
        distances = haversine_vectorise(lat[:-1], lon[:-1], lat[1:], lon[1:])

        # dt ≤ 0 → vitesse indéfinie (NaN). np.where évite la division
        # par zéro.
        delta_t_h = np.where(delta_t > 0, delta_t / 3600.0, np.nan)
        vitesses  = distances / delta_t_h   # km/h, NaN si dt ≤ 0

        # ── Niveau 1 : Z-score modifié ASYMÉTRIQUE ──────────────────
        # On calcule le MAD sur toutes les vitesses valides pour avoir
        # une référence robuste de la dispersion globale du segment.
        # Mais les CANDIDATS ne sont sélectionnés que parmi les vitesses
        # SUPÉRIEURES à la médiane (asymétrie).
        v_valides = vitesses[~np.isnan(vitesses)]
        if len(v_valides) > 0:
            v_med = float(np.median(v_valides))
            mad   = float(np.median(np.abs(v_valides - v_med)))
        else:
            v_med, mad = 0.0, 0.0

        # ── V5 : seuils relatifs (les max/min sont documentes) ─────
        #
        # (a) seuil_v_max : valeur la plus stricte entre le plafond
        #     absolu et un seuil contextuel. Plancher MAD_FALLBACK_SPEED
        #     pour eviter un seuil absurde sur segments quasi-
        #     stationnaires (v_med ~ 0).
        #     - marche (v_med=3)    : max(5, 20*3)=60 ; min(200,60)=60 ✓
        #     - bus   (v_med=30)    : max(5, 600)=600 ; min(200,600)=200
        #     - train (v_med=150)   : max(5, 3000)=3000 ; min(200,3000)=200
        #       (leger trade-off : un train a 250 km/h serait marque,
        #        mais c'est une limite acceptable de GeoLife)
        if v_med > 0:
            seuil_contextuel = max(MAD_FALLBACK_SPEED,
                                    MAX_SPEED_RELATIF * v_med)
            seuil_v_max = (min(max_speed_kmh, seuil_contextuel)
                           if max_speed_kmh is not None
                           else seuil_contextuel)
        else:
            # v_med = 0 : segment completement immobile. Pas de seuil
            # contextuel possible ; on retombe sur l'absolu pur (ou
            # infini si pas de plafond fourni).
            seuil_v_max = (max_speed_kmh
                           if max_speed_kmh is not None
                           else np.inf)

        # (b) seuil_persistance : max(plancher, fraction * N).
        #     - N=300  : 0.01*300 = 3   < 5  -> 5
        #     - N=3000 : 0.01*3000 = 30      -> 30
        #     Evite de marquer "regime" des clusters de 5 points sur
        #     5000 qui peuvent etre du bruit statistique.
        seuil_persistance = max(MAD_PERSISTANCE_MIN,
                                round(MAD_PERSISTANCE_FRACTION * n))

        # (c) seuil_dv : max(plancher, k * MAD_v).
        #     - regulier (MAD_v=0.5) : max(20, 2.5) = 20
        #     - hetero   (MAD_v=30)  : max(20, 150) = 150
        #     Laisser passer les transitions "normales pour ce segment".
        seuil_dv = max(MAD_DELTA_V_BRUTAL,
                       MAD_DELTA_V_FACTEUR_MAD * mad)

        # (d) seuil_fallback (MAD=0) : max(plancher, k * v_med).
        seuil_fallback = max(MAD_FALLBACK_SPEED,
                             MAD_FALLBACK_SPEED_FACTEUR * v_med)

        # ── Z-score asymetrique (V4, INCHANGE) ─────────────────────
        # MAD_Z_SEUIL = 3.5 reste absolu : c'est une reference
        # litteraire (Iglewicz & Hoaglin 1993), et le MAD fournit deja
        # l'echelle relative au segment.
        if mad > 0:
            z_scores = np.where(
                (~np.isnan(vitesses)) & (vitesses > v_med),
                0.6745 * (vitesses - v_med) / mad,
                0.0
            )
            candidats = z_scores > MAD_Z_SEUIL
        else:
            # MAD = 0 : fallback absolu sur seuil_fallback.
            # Asymetrie conservee : on ne marque que les points
            # strictement superieurs a la mediane.
            candidats = (
                (np.nan_to_num(vitesses, nan=0.0) > seuil_fallback)
                & (np.nan_to_num(vitesses, nan=0.0) > v_med)
            )

        # Seuil absolu de dernier recours (V5 : seuil_v_max, pas
        # max_speed_kmh directement).
        candidats = candidats | (
            np.nan_to_num(vitesses, nan=0.0) > seuil_v_max
        )

        # candidats est indexé sur les intervalles (longueur n-1).
        # On l'étend à la longueur n (indexé sur les points) :
        # l'intervalle (k-1, k) → point k.
        candidats_pts = np.concatenate([[False], candidats])

        # ── Niveau 2 : filtre contextuel ────────────────────────────
        # La boucle est intentionnellement séquentielle : l'analyse de
        # la persistance dépend des points précédents et ne se vectorise
        # pas sans perte de lisibilité.
        i = 1
        while i < n:
            if not candidats_pts[i]:
                i += 1
                continue

            v_i    = vitesses[i-1] if not np.isnan(vitesses[i-1]) else 0.0
            v_av   = (vitesses[i-2]
                      if (i >= 2 and not np.isnan(vitesses[i-2]))
                      else 0.0)
            dv     = abs(v_i - v_av)

            # Longueur du cluster consécutif de candidats
            j = i
            while j < n and candidats_pts[j]:
                j += 1
            persistance = j - i

            if persistance >= seuil_persistance:
                # Régime soutenu → MARQUER SEULEMENT LE PREMIER POINT
                # (correction seuil_dyn.pdf §2.3). Les points suivants
                # du cluster (i+1 à j-1) restent non marqués : ils ne
                # sont pas des artefacts mais des points d'un mode de
                # transport différent.
                suspect[i]      = True
                type_suspect[i] = "regime"
                i = j   # sauter tout le cluster

            elif dv > seuil_dv:
                # Variation brutale et isolée → artefact GPS ponctuel
                suspect[i]      = True
                type_suspect[i] = "artefact"
                i += 1

            else:
                # Bruit local modéré → non marqué
                i += 1

        # Seuil absolu de dernier recours en post-traitement
        # (V5 : seuil_v_max, pas max_speed_kmh).
        vitesses_pts = np.concatenate([[np.nan], vitesses])
        mask_absolu  = np.nan_to_num(vitesses_pts, nan=0.0) > seuil_v_max
        for i in range(n):
            if mask_absolu[i] and not suspect[i]:
                suspect[i]      = True
                type_suspect[i] = "artefact"

    df = df.copy()
    df["suspect_vitesse"]      = suspect
    df["type_suspect_vitesse"] = type_suspect

    # ── Niveau 3 : alerte globale d'hétérogénéité ───────────────────
    # V5 : MAD_R_TEST_NUL est prevu pour comparer au quantile 95% de
    # la distribution nulle (via validation/modeles_nuls.py). En
    # attendant l'implementation de cette etape ulterieure, on
    # conserve le seuil absolu MAD_R_HETEROGENE (comportement V4).
    r = (stats_segment or {}).get("ratio_heterogeneite")
    df["alerte_heterogene"] = bool(
        r is not None and r > MAD_R_HETEROGENE
    )

    return df