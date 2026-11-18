# ./pretraitement/courbe.py
#
# Trace, pour chaque segment, la série des distances inter-points en
# fonction du temps réel (secondes depuis le début du segment).
#
# Pourquoi le temps réel et non le numéro de point (cf. demande) :
# DFA, R/S et Whittle reçoivent la série d_i = haversine(p_{i-1}, p_i)
# indexée par le numéro de point -- mais visuellement, un gap de 200s
# et un pas de 5s ne méritent pas le même espace sur l'axe des abscisses.
# L'axe temporel réel révèle deux choses invisibles sur un axe en numéro
# de point : (1) les gaps internes (espace vide entre deux points),
# (2) l'irrégularité de l'échantillonnage (Δt variable).
#
# Ordonnée : d_i en mètres, la série brute que reçoivent les méthodes
# d'estimation -- pas la vitesse, pas le déplacement cumulé.
#
# Les points suspects (suspect_dist ou suspect_vitesse) sont marqués
# en rouge sur la courbe : cela permet de voir visuellement leur
# position dans la série et leur impact potentiel sur l'estimation de H.
#
# Sortie : log_segments/<user_id>/courbe_distance_<seg_id>.png
#
# ============================================================
# V5 (cf. documentation/CONSTANTES.md) — modifications apportees :
# ------------------------------------------------------------
# Le seuil "gap temporel notable" est maintenant calcule par le
# helper _seuil_gap_effectif(pas_median), qui applique la formule
# centralisee dans config.py :
#     seuil = max(SEUIL_GAP_NOTABLE_MIN,
#                 SEUIL_GAP_NOTABLE_RELATIF * pas_median)
#
# Avant (V4) : le facteur 2.5 etait hardcode dans deux endroits
# (tracer_courbe_segment et tracer_courbe_serie). Il est maintenant
# pilote par config.py, comme dans journal.py (etape 4).
#
# Impact visuel : PARITE V4 sur les segments GeoLife (2.5*7 = 17.5
# est deja > 5, donc le plancher ne joue pas). Effet reel sur les
# segments a tres petit pas (2 s) : seuil = max(5, 5) = 5, idem V4.
# Seul le cas tres petit pas + facteur modifie serait visible ; ce
# n'est PAS le cas par defaut. La centralisation est donc neutre
# aujourd'hui, mais prete pour une calibration future.
#
# Le titre de la courbe affiche desormais le pas median et le seuil
# de gap effectif si stats les fournit (utile pour la relecture).
# ============================================================

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")   # pas d'affichage, rendu fichier uniquement
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import (
    COURBE_DEUX_PANNEAUX,
    SEUIL_GAP_NOTABLE_MIN,
    SEUIL_GAP_NOTABLE_RELATIF,
)
from pretraitement.geodesie import haversine_vectorise


def _seuil_gap_effectif(pas_median):
    """
    Seuil de gap temporel effectif : max(plancher, k x pas median).

    Miroir exact de la formule utilisee dans journal.py
    (_seuil_gap_segment) : la valeur de reference vient de config.py
    (SEUIL_GAP_NOTABLE_MIN, SEUIL_GAP_NOTABLE_RELATIF). Ce helper
    evite de dupliquer la formule dans courbe.py, sans pour autant
    importer journal.py (couplage inutile : journal est un module
    plus lourd qui importe pandas via son DataFrame).

    Entree :
        pas_median : float, mediane des ecarts de temps (s)
                     (peut etre 0 ou NaN si segment degenere)

    Sortie :
        float : seuil de gap en secondes (>= SEUIL_GAP_NOTABLE_MIN).
    """
    if pas_median is None or not np.isfinite(pas_median) or pas_median <= 0:
        return float(SEUIL_GAP_NOTABLE_MIN)
    return max(float(SEUIL_GAP_NOTABLE_MIN),
               float(SEUIL_GAP_NOTABLE_RELATIF) * float(pas_median))


def tracer_courbe_segment(df, seg_id, dossier_sortie, stats=None, h_estimes=None,
                          panneau_indice=COURBE_DEUX_PANNEAUX):
    """
    Trace la série des distances inter-points d_i = haversine(p_{i-1}, p_i)
    en fonction du temps réel depuis le début du segment, et sauvegarde
    l'image en PNG.

    Entrée :
        df            : DataFrame du segment (colonnes "lat", "lon",
                        "timestamp", optionnellement "suspect_dist" et
                        "suspect_vitesse" issues de la Phase 3)
        seg_id        : int, identifiant du segment (pour le nom du fichier)
        dossier_sortie: str, dossier où écrire l'image (même dossier que
                        les fichiers log_segment_X.txt)
        stats         : dict ou None, statistiques Phase 2
                        (calculer_stats_segment()) -- affichées dans le
                        titre si fournies (V5 : inclut pas_median_s et
                        seuil_gap effectif)
        h_estimes     : dict ou None (V5), ex. {"DFA": 0.82, "R/S": 0.78,
                        "Whittle": 0.85} -- ajoute une ligne "H estimés" au
                        titre pour comparer avec la lecture à l'œil
        panneau_indice: bool (V5), si True ajoute un second panneau : d_i en
                        fonction du NUMÉRO de point, c'est-à-dire exactement la
                        série que voient DFA, R/S et Whittle (le premier panneau
                        est en temps réel). Défaut : config.COURBE_DEUX_PANNEAUX

    Sortie :
        str : chemin de l'image PNG créée
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    n = len(df)
    if n < 2:
        return None   # rien à tracer pour un segment d'un seul point

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    t   = df["timestamp"].to_numpy()

    # --- Axe des abscisses : temps réel depuis le début du segment ---
    # t[0] = timestamp du premier point ; t_rel[i] = secondes écoulées
    # depuis ce point pour chaque paire (i-1, i). Le premier point de la
    # série de distances correspond à la paire (0,1), donc t_rel[0] = t[1]-t[0].
    t_rel = t[1:] - t[0]   # tableau de longueur n-1

    # --- Ordonnée : distances inter-points en mètres ---
    # haversine_vectorise() retourne des km ; on multiplie par 1000.
    distances_m = haversine_vectorise(lat[:-1], lon[:-1], lat[1:], lon[1:]) * 1000.0

    # --- Masques des points suspects (Phase 3) ---
    has_suspect = ("suspect_dist" in df.columns or "suspect_vitesse" in df.columns)
    if has_suspect:
        # Les colonnes de marquage portent sur le point i, pas sur
        # l'intervalle (i-1, i) -- on associe chaque distance d_i au
        # marquage du point i (indice 1..n-1 dans le DataFrame).
        suspect = np.zeros(n-1, dtype=bool)
        if "suspect_dist" in df.columns:
            suspect |= df["suspect_dist"].to_numpy()[1:]
        if "suspect_vitesse" in df.columns:
            suspect |= df["suspect_vitesse"].to_numpy()[1:]

    # --- Figure ---
    if panneau_indice:
        fig, (ax, ax_indice) = plt.subplots(2, 1, figsize=(12, 7))
    else:
        fig, ax = plt.subplots(figsize=(12, 4))

    # Ligne de base : tous les points
    ax.plot(t_rel, distances_m, color="#4a90d9", linewidth=0.7,
            alpha=0.85, label="distance inter-points (m)", zorder=2)

    # Points suspects mis en évidence
    if has_suspect and suspect.any():
        ax.scatter(t_rel[suspect], distances_m[suspect],
                   color="#e74c3c", s=40, zorder=3,
                   label=f"point suspect ({suspect.sum()})")

    # Ligne médiane (référence visuelle)
    mediane = float(np.median(distances_m))
    ax.axhline(mediane, color="#888888", linewidth=0.8, linestyle="--",
               label=f"médiane = {mediane:.1f} m", zorder=1)

    # --- Gaps temporels : zones grisées ---
    # V5 : seuil RELATIF calcule par _seuil_gap_effectif (formule
    # centralisee dans config.py, partagee avec journal.py).
    # Un gap est un intervalle dont le delta_t depasse ce seuil.
    # On les matérialise par une zone verticale grisée pour que l'œil
    # perçoive visuellement l'irrégularité de l'échantillonnage.
    dt_local = np.diff(t)   # longueur n-1
    dt_median_local = float(np.median(dt_local)) if len(dt_local) > 0 else 0.0
    seuil_gap_local = _seuil_gap_effectif(dt_median_local)
    gaps_mask = dt_local > seuil_gap_local

    for i, is_gap in enumerate(gaps_mask):
        if is_gap:
            # le gap s'étend de t_rel[i-1] à t_rel[i]
            x_gauche = t_rel[i-1] if i > 0 else 0
            x_droite = t_rel[i]
            ax.axvspan(x_gauche, x_droite, alpha=0.12, color="#f39c12",
                       zorder=0, label="_nolegend_")

    # Légende unique pour les zones de gap (une seule entrée même si
    # plusieurs gaps). V5 : le seuil effectif est affiche pour rendre
    # l'echelle temporelle du gap lisible a l'oeil.
    if gaps_mask.any():
        from matplotlib.patches import Patch
        patch_gap = Patch(facecolor="#f39c12", alpha=0.25,
                          label=f"gap temporel notable "
                                f"({gaps_mask.sum()}, seuil={seuil_gap_local:.1f}s)")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(patch_gap)
        ax.legend(handles=handles, fontsize=8, loc="upper right")
    else:
        ax.legend(fontsize=8, loc="upper right")

    # --- Titre ---
    debut_heure = _ts_to_hms(t[0])
    duree_s     = t[-1] - t[0]
    if stats and stats.get("mediane_vitesse") is not None:
        titre = (f"Segment {seg_id} — {n} points — début {debut_heure} — "
                 f"durée {duree_s/60:.1f} min\n"
                 f"v_mèd={stats['mediane_vitesse']:.1f} km/h  "
                 f"d_mèd={stats['distance_mediane']*1000:.1f} m  "
                 f"d_max={stats['distance_max']*1000:.1f} m")
    else:
        titre = f"Segment {seg_id} — {n} points — début {debut_heure} — durée {duree_s/60:.1f} min"

    # V5 : pas median + seuil de gap effectif (lisibilite a froid).
    if stats and stats.get("pas_median_s") is not None:
        titre += (f"\npas_médian={stats['pas_median_s']:.2f}s · "
                  f"seuil_gap={seuil_gap_local:.1f}s")

    if h_estimes:
        titre += "\nH estimés : " + "   ".join(
            f"{nom}={val:.2f}" for nom, val in h_estimes.items() if val is not None)
    ax.set_title(titre, fontsize=9, pad=8)

    # --- Axes ---
    ax.set_xlabel("Temps depuis le début du segment (s)", fontsize=9)
    ax.set_ylabel("Distance inter-points (m)", fontsize=9)
    ax.tick_params(labelsize=8)

    # Formatter l'axe X en mm:ss si la durée le justifie
    if duree_s <= 3600:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x//60):02d}:{int(x%60):02d}"
        ))
        ax.set_xlabel("Temps depuis le début du segment (mm:ss)", fontsize=9)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    if panneau_indice:
        # Second panneau : la serie telle que les estimateurs la voient (axe = numéro de point)
        ax_indice.plot(np.arange(1, n), distances_m, color="#4a90d9", linewidth=0.7, alpha=0.85)
        if has_suspect and suspect.any():
            ax_indice.scatter(np.arange(1, n)[suspect], distances_m[suspect], color="#e74c3c", s=40, zorder=3)
        ax_indice.axhline(mediane, color="#888888", linewidth=0.8, linestyle="--")
        ax_indice.set_xlabel("Numéro du point (axe utilisé par DFA, R/S et Whittle)", fontsize=9)
        ax_indice.set_ylabel("Distance inter-points (m)", fontsize=9)
        ax_indice.tick_params(labelsize=8)
        ax_indice.set_xlim(left=1)
        ax_indice.set_ylim(bottom=0)
        ax_indice.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.tight_layout()

    chemin = os.path.join(dossier_sortie, f"courbe_distance_{seg_id}.png")
    fig.savefig(chemin, dpi=130)
    plt.close(fig)

    return chemin


def tracer_courbe_serie(t_rel, valeurs, chemin, titre="", ligne_mediane=True):
    """
    Trace une serie quelconque (reelle ou simulee) dans le MEME style que tracer_courbe_segment :
    ligne bleue, mediane en pointille, zones de gap grisees. Aucune statistique, aucun H, aucun
    point rouge : sert a la validation visuelle a l'aveugle (experiences/validation_visuelle.py),
    ou l'image ne doit rien reveler.

    Entree :
        t_rel   : abscisses en secondes depuis le debut (longueur = len(valeurs))
        valeurs : ordonnees (distances en metres)
        chemin  : fichier PNG a ecrire
        titre   : titre (vide par defaut)
    Sortie : chemin
    """
    t_rel = np.asarray(t_rel, float)
    valeurs = np.asarray(valeurs, float)
    os.makedirs(os.path.dirname(chemin) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_rel, valeurs, color="#4a90d9", linewidth=0.7, alpha=0.85, zorder=2)
    if ligne_mediane:
        ax.axhline(float(np.median(valeurs)), color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    pas = np.diff(t_rel)
    if len(pas) > 0:
        # V5 : seuil de gap calcule par _seuil_gap_effectif (parite avec
        # tracer_courbe_segment et journal.py). Aucune fuite
        # d'information : le seuil est identique pour vraie/fausse serie.
        seuil_gap = _seuil_gap_effectif(float(np.median(pas)))
        gaps = pas > seuil_gap
        for i in np.flatnonzero(gaps):
            ax.axvspan(t_rel[i], t_rel[i + 1], alpha=0.12, color="#f39c12", zorder=0)
    if titre:
        ax.set_title(titre, fontsize=9, pad=8)
    ax.set_xlabel("Temps depuis le début du segment (s)", fontsize=9)
    ax.set_ylabel("Distance inter-points (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(chemin, dpi=130)
    plt.close(fig)
    return chemin


def _ts_to_hms(timestamp):
    """Convertit un timestamp UNIX en chaîne HH:MM:SS."""
    from datetime import datetime
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


def tracer_courbes_utilisateur(user_id, segments_info, dossier_log_racine):
    """
    Appelle tracer_courbe_segment() pour chaque segment d'un utilisateur
    et retourne la liste des fichiers PNG créés.

    Entrée :
        user_id            : str (ex: "003")
        segments_info      : list of dict, même structure qu'attendue
                             par journal.ecrire_log_utilisateur() --
                             clés "df", "stats", "retenu"
        dossier_log_racine : str (ex: "log_segments")

    Sortie :
        list of str : chemins des PNG créés (None ignorés)
    """
    dossier_user = os.path.join(dossier_log_racine, user_id)
    chemins = []

    for seg_id, info in enumerate(segments_info):
        chemin = tracer_courbe_segment(
            df=info["df"],
            seg_id=seg_id,
            dossier_sortie=dossier_user,
            stats=info.get("stats"),
        )
        if chemin:
            chemins.append(chemin)

    return chemins