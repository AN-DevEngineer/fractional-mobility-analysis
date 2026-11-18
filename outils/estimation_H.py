import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # racine du projet : permet `python outils/<script>.py`
import os
import csv

from hurst.dfa import DFA
from hurst.rs import RS
from hurst.whittle import Whittle
from hurst.statistiques_segment import statistiques_segment, afficher_statistiques


# Nombre minimum de points pour retenter une estimation de H sur la
# version "sans points suspects" d'un segment (en deça, DFA/RS/Whittle
# ne sont pas fiables — meme seuil d'esprit que N_MIN_WHITTLE de
# config.py).
MIN_POINTS_H_SANS_SUSPECTS = 20

# Seuil au-dela duquel un ecart entre H (tous points) et H (sans les
# points suspects) est signale comme potentiellement problematique :
# cela indiquerait qu'un ou deux points suspects pesent fortement sur
# l'estimation, et que la question encore ouverte dans les documents
# de reflexion ("faut-il couper ?") devrait probablement se trancher
# en faveur de la coupure pour ce segment.
SEUIL_ECART_H_SUSPECT = 0.10


def lire_fichier_csv(file_path):
    """
    Lit le CSV produit par pretraitement/sauvegarde.py

    Colonnes attendues (lues par NOM, pas par position, pour rester
    robuste a un changement d'ordre des colonnes) :
        segment_id, point_id, lat, lon, alt, timestamp,
        suspect_dist, suspect_vitesse

    Les deux dernieres colonnes sont OPTIONNELLES : si le CSV ne les
    contient pas (segments non marques), elles sont considerees a
    False par defaut.

    Entree :
        file_path (str) : chemin du CSV (ex: "./sortie/001.csv")

    Sortie :
        list of dict : un dict par point, avec les cles
            "segment_id", "lat", "lon", "timestamp",
            "suspect_dist", "suspect_vitesse"
        Retourne [] si le fichier est illisible.
    """
    points = []

    try:
        f = open(file_path, "r", newline="")
    except OSError as e:
        print(f"[ERREUR] Impossible d'ouvrir : {file_path} → {e}")
        return points

    with f:
        lecteur = csv.DictReader(f)

        for numero_ligne, ligne in enumerate(lecteur, start=2):
            try:
                segment_id = int(float(ligne["segment_id"]))
                lat        = float(ligne["lat"])
                lon        = float(ligne["lon"])
                timestamp  = float(ligne["timestamp"])

                # .get() : colonnes absentes -> "False" par defaut
                # (segments produits avant l'ajout du marquage, ou CSV
                # externe sans ces colonnes).
                suspect_dist    = ligne.get("suspect_dist", "False").strip().lower() == "true"
                suspect_vitesse = ligne.get("suspect_vitesse", "False").strip().lower() == "true"

            except (ValueError, KeyError, AttributeError) as e:
                print(f"[AVERTISSEMENT] Ligne {numero_ligne} non parseable dans {file_path} → {e}")
                continue

            points.append({
                "segment_id":      segment_id,
                "lat":             lat,
                "lon":             lon,
                "timestamp":       timestamp,
                "suspect_dist":    suspect_dist,
                "suspect_vitesse": suspect_vitesse,
            })

    return points


def reconstruire_segments(points):
    """
    Regroupe les points par segment_id, dans l'ordre d'apparition

    Corrige le bogue de l'ancienne implementation (boucle "if
    Points[i][0] == num_seg ... else ...") qui perdait le premier
    point de chaque segment sauf le premier : ici, le regroupement se
    fait en detectant un changement de segment_id, et le point qui
    declenche ce changement est bien ajoute au NOUVEAU segment, pas
    perdu. Ne suppose plus non plus que les segment_id sont contigus
    a partir de 0 — seul l'ordre d'apparition compte.

    Pour chaque segment, deux listes de points GPS [lat, lon,
    timestamp] sont produites : tous les points, et seulement ceux
    qui ne sont marques NI suspect_dist NI suspect_vitesse. Cela
    permet ensuite de comparer l'estimation de H avec et sans les
    points suspects (cf. main, plus bas) — l'objectif declare du
    marquage en Phase 3 du pipeline plutot que de la coupure.

    Entree :
        points : list of dict, sortie de lire_fichier_csv()

    Sortie :
        list of dict, un par segment, avec les cles :
            "segment_id"      : identifiant d'origine
            "points_complets" : [[lat, lon, timestamp], ...]
            "points_filtres"  : [[lat, lon, timestamp], ...]
                                 (sous-ensemble sans points suspects)
    """
    segments        = []
    segment_complet = []
    segment_filtre  = []
    seg_id_courant  = None

    for p in points:
        if p["segment_id"] != seg_id_courant:
            if segment_complet:
                segments.append({
                    "segment_id":      seg_id_courant,
                    "points_complets": segment_complet,
                    "points_filtres":  segment_filtre,
                })
            segment_complet = []
            segment_filtre  = []
            seg_id_courant  = p["segment_id"]

        pt = [p["lat"], p["lon"], p["timestamp"]]
        segment_complet.append(pt)

        if not (p["suspect_dist"] or p["suspect_vitesse"]):
            segment_filtre.append(pt)

    if segment_complet:
        segments.append({
            "segment_id":      seg_id_courant,
            "points_complets": segment_complet,
            "points_filtres":  segment_filtre,
        })

    return segments


def sauvegarder_resultats_H(user_id, lignes, dossier="resultats_H"):
    """
    Sauvegarde les résultats H dans resultats_H/<user_id>_H.csv

    Colonnes :
        segment_id, n_points, h_dfa, h_rs, h_whittle,
        distance_max_m, vitesse_max_kmh, delta_t_moyen_s, divergence,
        n_points_suspects, h_dfa_sans_suspects, ecart_h_dfa_suspects
    """
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, f"{user_id}_H.csv")

    with open(chemin, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_id",
            "n_points",
            "h_dfa",
            "h_rs",
            "h_whittle",
            "distance_max_m",
            "vitesse_max_kmh",
            "delta_t_moyen_s",
            "divergence",
            "n_points_suspects",
            "h_dfa_sans_suspects",
            "ecart_h_dfa_suspects",
        ])
        writer.writerows(lignes)

    print(f"\nRésultats sauvegardés : {chemin}")


def calculer_divergence(h_dfa, h_rs, h_whittle):
    """
    Écart entre la valeur max et min parmi les estimations disponibles.
    Retourne None si moins de 2 valeurs valides.
    """
    valeurs = [h for h in [h_dfa, h_rs, h_whittle] if h is not None]

    if len(valeurs) < 2:
        return None

    return round(max(valeurs) - min(valeurs), 7)


# ── Chargement ──────────────────────────────────────────────

chemin  = './sortie/182.csv'
user_id = os.path.splitext(os.path.basename(chemin))[0]  # "000"

Points = lire_fichier_csv(chemin)

# ── Reconstruction des segments ─────────────────────────────

Segments = reconstruire_segments(Points)

# ── Analyse ─────────────────────────────────────────────────

print(f"Utilisateur {user_id}  —  {len(Segments)} segments")
print()

lignes_csv = []

for j, seg in enumerate(Segments):

    points_complets = seg["points_complets"]
    points_filtres  = seg["points_filtres"]
    nb_suspects     = len(points_complets) - len(points_filtres)

    H_dfa      = DFA(points_complets)
    H_rs       = RS(points_complets)
    H_whittle  = Whittle(points_complets)
    stats      = statistiques_segment(points_complets)
    divergence = calculer_divergence(H_dfa, H_rs, H_whittle)

    print(f"Segment {j} : H (DFA) = {H_dfa}")
    print(f"Segment {j} : H (RS)  = {H_rs}")
    print(f"Segment {j} : H (WHITTLE) = {H_whittle}")
    afficher_statistiques(j, H_dfa, stats)

    # Indicateur de qualité (divergence inter-méthodes)
    if divergence is not None:
        flag = " ⚠ SUSPECT" if divergence > 0.15 else " ✓"
        print(f"Segment {j} : divergence inter-méthodes = {divergence}{flag}")

    # --- Impact des points marqués suspects sur H (Phase 3 du pipeline) ---
    # C'est ici que le marquage (plutôt que la coupure) prend tout son
    # sens : on peut observer si H change significativement une fois
    # les points suspects retirés, avant de décider de les couper pour
    # de bon dans le pipeline.
    H_dfa_sans_suspects  = None
    ecart_H_dfa_suspects = None

    if nb_suspects > 0:
        if len(points_filtres) >= MIN_POINTS_H_SANS_SUSPECTS:
            H_dfa_sans_suspects = DFA(points_filtres)
            if H_dfa_sans_suspects is not None and H_dfa is not None:
                ecart_H_dfa_suspects = round(abs(H_dfa - H_dfa_sans_suspects), 7)

            print(f"Segment {j} : {nb_suspects} point(s) suspect(s) "
                  f"({len(points_complets)} -> {len(points_filtres)} points)")
            print(f"Segment {j} : H (DFA) sans points suspects = {H_dfa_sans_suspects}")
            if ecart_H_dfa_suspects is not None:
                flag = " ⚠ DEPENDANT DES POINTS SUSPECTS" if ecart_H_dfa_suspects > SEUIL_ECART_H_SUSPECT else " ✓"
                print(f"Segment {j} : écart H (avec/sans suspects) = {ecart_H_dfa_suspects}{flag}")
        else:
            print(f"Segment {j} : {nb_suspects} point(s) suspect(s), mais "
                  f"trop peu de points restants ({len(points_filtres)}) "
                  f"pour ré-estimer H sans eux")

    print()

    lignes_csv.append([
        j,
        stats.get("nb_points"),
        H_dfa,
        H_rs,
        H_whittle,
        stats.get("dist_max"),
        stats.get("vitesse_max_kmh"),
        stats.get("delta_t_moyen_s"),
        divergence,
        nb_suspects,
        H_dfa_sans_suspects,
        ecart_H_dfa_suspects,
    ])

# ── Sauvegarde ───────────────────────────────────────────────

sauvegarder_resultats_H(user_id, lignes_csv)