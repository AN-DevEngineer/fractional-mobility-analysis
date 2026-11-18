#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_H_valides.py

Post-traitement de resultats_H/<user>_H.csv (AUCUNE estimation recalculee).

Produit, dans resultats_H_valides/ :
  - <user>_H_gardes.csv : une ligne par segment GARDE + colonne fort_0p5
  - synthese_H.csv      : une ligne par utilisateur, 3 scenarios

Critere d'inclusion d'un segment (sinon exclu) :
  - h_dfa, h_rs, h_whittle tous calcules
  - 0 < h_dfa < 1  ET  0 < h_rs < 1  ET  0 < h_whittle < 1
  - whittle_a_la_borne == False (strictement renseigne, pas vide)
  - divergence <= seuil (defaut 0.15)

Scenario "forts" : sous-ensemble des gardes ou les 3 methodes donnent H > 0.5.

═══════════════════════════════════════════════════════════════════
LIMITES METHODOLOGIQUES ASSUMEES (a documenter dans le memoire)
═══════════════════════════════════════════════════════════════════
1. BIAIS WHITTLE : la divergence est calculee avec h_whittle BRUT.
   Whittle local (Robinson 1995) est biaise vers le haut sur series
   courtes (biais ~+0.05 a +0.15 selon N). Ce biais gonfle la
   divergence max-min et exclut potentiellement des segments valides.
   Correction possible : Shimotsu (2009) exact local Whittle, non
   implemente ici. Voir aussi Arteche & Martins (2024) TVLW.

2. WHITTLE SATURE : si whittle_a_la_borne est vide (""), le segment
   est EXCLU. Un Whittle vide = Whittle non calcule = segment non
   exploitable pour le filtre inter-methodes.

3. PONDERATION wavg : h_dfa_wavg_tous pondere par n_points. Sur les
   longs segments (contamines par les regimes), cela SURESTIME encore
   h_dfa_tous. La wavg n'est fournie que pour tracer l'effet de la
   ponderation ; h_dfa_gardes (arithmetique) reste la metrique
   principale. h_dfa_wavg_gardes est informative (segments propres).

4. SCENARIO FORTS : tres proche de GARDES par construction. Si
   n_forts ~= n_gardes, le scenario forts n'apporte pas d'information
   supplementaire (c'est un resultat : le filtre selectionne deja
   massivement des segments fortement persistants).

5. PAS DE REPRISE : le script reconstruit tout a chaque execution
   (duree < 1 min). Changer --seuil-divergence entre deux runs ne
   peut pas produire un CSV incoherent.

Options :
  --source DOSSIER       (defaut : resultats_H)
  --sortie DOSSIER       (defaut : resultats_H_valides)
  --seuil-divergence F   (defaut : 0.15)
  --seuil-fort F         (defaut : 0.5)
"""

import os
import csv
import argparse
import math
from datetime import datetime

# ============================================================
# CONSTANTES
# ============================================================

SOURCE_DIR         = "resultats_H"
SORTIE_DIR         = "resultats_H_valides"
SEUIL_DIVERGENCE   = 0.15
SEUIL_FORT         = 0.5

COLONNES_GARDES = [
    "segment_id", "n_points",
    "h_dfa", "h_rs", "h_whittle", "divergence",
    "h_rs_corrige", "dfa_bas", "dfa_haut", "whittle_m",
    "nul_p", "nul_ordre_ar",
    "mode_echantillonnage", "pas_median_s",
    "cv_d", "part_immobile", "rapport_net", "duree_s", "trajet_m",
    "h_dfa_morceaux", "n_morceaux",
    "fort_0p5", "empreinte_config", "serie",
]

COLONNES_SYNTHESE = [
    "user_id",
    "n_segments_total", "n_segments_gardes", "n_segments_forts",
    "h_dfa_tous",   "h_rs_tous",   "h_wh_tous",   "h_dfa_wavg_tous",
    "h_dfa_gardes", "h_rs_gardes", "h_wh_gardes", "h_dfa_wavg_gardes",
    "h_dfa_forts",  "h_rs_forts",  "h_wh_forts",  "h_dfa_wavg_forts",
]


# ============================================================
# HELPERS
# ============================================================

def _float_or_none(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v != v else v


def _int_or_none(x):
    v = _float_or_none(x)
    return None if v is None else int(round(v))


def _parse_bool(x):
    """True/False strict. Retourne None si vide (cas a distinguer)."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s.lower() in ("true", "1", "yes", "vrai", "o", "v")


def _moy(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def _moy_pond(vals, poids):
    couples = [(v, p) for v, p in zip(vals, poids) if v is not None and p]
    if not couples:
        return None
    return round(sum(v * p for v, p in couples) / sum(p for _, p in couples), 6)


def _quantile(vals, q):
    """
    Quantile empirique par interpolation lineaire, sans numpy.
    q in [0, 1]. Retourne None si vals vide.
    """
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def _histogramme_texte(vals, bins=10, largeur=40):
    """
    Retourne une liste de lignes ASCII representant l'histogramme.
    Affiche min, max, et N par bin.
    """
    vals = [v for v in vals if v is not None]
    if not vals:
        return ["  (aucune valeur)"]
    vmin, vmax = min(vals), max(vals)
    if vmin == vmax:
        return [f"  Tous a {vmin:.3f} (n={len(vals)})"]
    largeur_bin = (vmax - vmin) / bins
    comptes = [0] * bins
    for v in vals:
        idx = min(int((v - vmin) / largeur_bin), bins - 1)
        comptes[idx] += 1
    n_max = max(comptes) if comptes else 1
    lignes = []
    for i, c in enumerate(comptes):
        lo = vmin + i * largeur_bin
        hi = lo + largeur_bin
        barre = "#" * int((c / n_max) * largeur) if n_max else ""
        lignes.append(f"  [{lo:.3f}, {hi:.3f}) : {c:>4}  {barre}")
    return lignes


def _format_pour_csv(v):
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    return v


# ============================================================
# TRAITEMENT D'UN UTILISATEUR
# ============================================================

def traiter_utilisateur(user_id, chemin_source, dossier_sortie,
                        seuil_divergence, seuil_fort):
    """
    Lit <user>_H.csv, ecrit <user>_H_gardes.csv, retourne la ligne de synthese.
    Retourne None si la source est illisible ou vide.
    """
    try:
        with open(chemin_source, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        print(f"[ERREUR] {chemin_source} : {e}")
        return None

    if not rows:
        print(f"[INFO] {user_id} : aucun segment dans la source.")
        return None

    tous_h_dfa, tous_h_rs, tous_h_wh, tous_poids = [], [], [], []
    gardes = []
    n_wh_borne_vide = 0  # compteur pour diagnostic

    for row in rows:
        h_dfa = _float_or_none(row.get("h_dfa"))
        h_rs  = _float_or_none(row.get("h_rs"))
        h_wh  = _float_or_none(row.get("h_whittle"))
        div   = _float_or_none(row.get("divergence"))
        n_pts = _int_or_none(row.get("n_points"))
        wh_borne = _parse_bool(row.get("whittle_a_la_borne"))

        # Accumulateurs "tous"
        if h_dfa is not None:
            tous_h_dfa.append(h_dfa)
            tous_poids.append(n_pts if n_pts else 0)
        if h_rs is not None:
            tous_h_rs.append(h_rs)
        if h_wh is not None:
            tous_h_wh.append(h_wh)

        # CORRECTIF : wh_borne doit etre explicitement False (pas None/vide)
        if wh_borne is None:
            n_wh_borne_vide += 1
            continue  # Whittle non calcule -> segment non exploitable

        # Critere d'inclusion
        if not (h_dfa is not None and h_rs is not None and h_wh is not None
                and 0 < h_dfa < 1 and 0 < h_rs < 1 and 0 < h_wh < 1
                and wh_borne is False
                and div is not None and div <= seuil_divergence):
            continue

        fort = 1 if (h_dfa > seuil_fort and h_rs > seuil_fort
                     and h_wh > seuil_fort) else 0
        gardes.append({**row, "fort_0p5": fort})

    if n_wh_borne_vide > 0:
        print(f"  [diag] {user_id} : {n_wh_borne_vide} segments exclus "
              f"(whittle_a_la_borne vide)")

    # --- Ecriture <user>_H_gardes.csv ---
    chemin_gardes = os.path.join(dossier_sortie, f"{user_id}_H_gardes.csv")
    with open(chemin_gardes, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLONNES_GARDES)
        for row in gardes:
            writer.writerow([_format_pour_csv(row.get(c)) for c in COLONNES_GARDES])

    # --- Metriques "gardes" et "forts" ---
    n_gardes = len(gardes)
    n_forts  = sum(1 for row in gardes if row["fort_0p5"] == 1)
    forts_rows = [row for row in gardes if row["fort_0p5"] == 1]

    def _cols(rows_):
        dfa   = [_float_or_none(r.get("h_dfa"))     for r in rows_]
        rs    = [_float_or_none(r.get("h_rs"))      for r in rows_]
        wh    = [_float_or_none(r.get("h_whittle")) for r in rows_]
        poids = [_int_or_none(r.get("n_points"))    for r in rows_]
        return dfa, rs, wh, poids

    g_dfa, g_rs, g_wh, g_poids = _cols(gardes)
    f_dfa, f_rs, f_wh, f_poids = _cols(forts_rows)

    ligne = {
        "user_id": user_id,
        "n_segments_total":  len(rows),
        "n_segments_gardes": n_gardes,
        "n_segments_forts":  n_forts,
        "h_dfa_tous":        _moy(tous_h_dfa),
        "h_rs_tous":         _moy(tous_h_rs),
        "h_wh_tous":         _moy(tous_h_wh),
        "h_dfa_wavg_tous":   _moy_pond(tous_h_dfa, tous_poids),
        "h_dfa_gardes":      _moy(g_dfa),
        "h_rs_gardes":       _moy(g_rs),
        "h_wh_gardes":       _moy(g_wh),
        "h_dfa_wavg_gardes": _moy_pond(g_dfa, g_poids),
        "h_dfa_forts":       _moy(f_dfa),
        "h_rs_forts":        _moy(f_rs),
        "h_wh_forts":        _moy(f_wh),
        "h_dfa_wavg_forts":  _moy_pond(f_dfa, f_poids),
    }
    # On garde les listes pour les stats globales
    ligne["_g_dfa"] = g_dfa
    ligne["_g_poids"] = g_poids
    return ligne


# ============================================================
# STATISTIQUES GLOBALES
# ============================================================

def afficher_stats_globales(lignes):
    """
    Calcule et affiche :
      - Taux d'exclusion global
      - Distribution de h_dfa_gardes (quantiles + histogramme)
      - Moyenne ponderee par n_segments_gardes (LE chiffre a citer)
      - Utilisateurs sans aucun segment garde
      - Proportion n_forts == n_gardes
    """
    print()
    print("=" * 72)
    print("STATISTIQUES GLOBALES")
    print("=" * 72)

    # Effectifs
    total_s = sum(l["n_segments_total"] for l in lignes)
    total_g = sum(l["n_segments_gardes"] for l in lignes)
    total_f = sum(l["n_segments_forts"]  for l in lignes)
    n_users = len(lignes)

    if total_s > 0:
        print(f"  Utilisateurs traites : {n_users}")
        print(f"  Segments totaux      : {total_s}")
        print(f"  Segments gardes      : {total_g}  "
              f"({100*total_g/total_s:.1f}%)")
        print(f"  Segments forts       : {total_f}  "
              f"({100*total_f/total_s:.1f}%)")
        print(f"  Taux d'exclusion     : "
              f"{100*(1 - total_g/total_s):.1f}%")
        print()

    # Moyenne ponderee par n_segments_gardes : LE chiffre de reference
    h_gardes_vals = [l["h_dfa_gardes"] for l in lignes
                     if l["h_dfa_gardes"] is not None]
    h_gardes_poids = [l["n_segments_gardes"] for l in lignes
                      if l["h_dfa_gardes"] is not None]
    h_global_pond = _moy_pond(h_gardes_vals, h_gardes_poids)
    h_global_arith = _moy(h_gardes_vals)

    if h_global_pond is not None:
        print(f"  H_DFA global (pondere n_gardes) : {h_global_pond:.4f}")
        print(f"  H_DFA global (arithmetique)     : {h_global_arith:.4f}")
        print()

    # Distribution de h_dfa_gardes (par utilisateur)
    print("  Distribution de h_dfa_gardes (un point par utilisateur) :")
    print(f"    n     : {len(h_gardes_vals)}")
    if h_gardes_vals:
        for q in (0.05, 0.25, 0.50, 0.75, 0.95):
            v = _quantile(h_gardes_vals, q)
            print(f"    q{q:.2f} : {v:.4f}" if v is not None else f"    q{q:.2f} : -")
        print()
        for ligne in _histogramme_texte(h_gardes_vals, bins=12, largeur=35):
            print(ligne)
    print()

    # Utilisateurs sans aucun segment garde
    users_0 = [l["user_id"] for l in lignes if l["n_segments_gardes"] == 0]
    if users_0:
        print(f"  {len(users_0)} utilisateur(s) sans aucun segment garde :")
        print(f"    {', '.join(users_0)}")
    else:
        print("  Tous les utilisateurs ont au moins 1 segment garde.")
    print()

    # Proportion n_forts == n_gardes
    n_egaux = sum(1 for l in lignes
                  if l["n_segments_forts"] == l["n_segments_gardes"]
                  and l["n_segments_gardes"] > 0)
    n_avec_gardes = sum(1 for l in lignes if l["n_segments_gardes"] > 0)
    if n_avec_gardes > 0:
        pct = 100 * n_egaux / n_avec_gardes
        print(f"  Utilisateurs avec n_forts = n_gardes : "
              f"{n_egaux}/{n_avec_gardes} ({pct:.1f}%)")
        if pct > 90:
            print("    -> Le scenario FORTS est quasi identique a GARDES. "
                  "A documenter comme resultat, pas comme scenario independant.")

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main(source_dir, sortie_dir, seuil_div, seuil_fort):
    if not os.path.isdir(source_dir):
        print(f"[ERREUR] Dossier source introuvable : {source_dir}")
        return
    os.makedirs(sortie_dir, exist_ok=True)

    csv_files = sorted(
        f for f in os.listdir(source_dir)
        if f.endswith("_H.csv") and not f.startswith("synthese")
    )
    print(f"Fichiers source    : {len(csv_files)} dans {source_dir}/")
    print(f"Sortie             : {sortie_dir}/")
    print(f"Seuil divergence   : {seuil_div}")
    print(f"Seuil fort         : {seuil_fort}")
    print()

    chemin_synthese = os.path.join(sortie_dir, "synthese_H.csv")

    # PAS DE REPRISE : on ecrase toujours
    f_syn = open(chemin_synthese, "w", newline="")
    writer_syn = csv.writer(f_syn)
    writer_syn.writerow(COLONNES_SYNTHESE)

    debut = datetime.now()
    n_traites = 0
    total_s = total_g = total_f = 0
    toutes_lignes = []  # pour les stats globales

    try:
        for csv_file in csv_files:
            user_id = csv_file[:-len("_H.csv")]
            ligne = traiter_utilisateur(
                user_id,
                os.path.join(source_dir, csv_file),
                sortie_dir,
                seuil_div,
                seuil_fort,
            )
            if ligne is None:
                continue

            writer_syn.writerow(
                [_format_pour_csv(ligne.get(c)) for c in COLONNES_SYNTHESE])
            f_syn.flush()

            toutes_lignes.append(ligne)
            n_traites += 1
            total_s   += ligne["n_segments_total"]
            total_g   += ligne["n_segments_gardes"]
            total_f   += ligne["n_segments_forts"]

            print(f"  {user_id:<10} | "
                  f"{ligne['n_segments_total']:>4} seg. | "
                  f"{ligne['n_segments_gardes']:>3} gardes | "
                  f"{ligne['n_segments_forts']:>3} forts  ||  "
                  f"G: DFA={ligne['h_dfa_gardes']}  "
                  f"RS={ligne['h_rs_gardes']}  "
                  f"WH={ligne['h_wh_gardes']}")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C : arret propre. Synthese partielle conservee.")
    finally:
        f_syn.close()

    duree = (datetime.now() - debut).total_seconds()
    print()
    print("=" * 72)
    print("BILAN RAPIDE")
    print("=" * 72)
    print(f"  Duree            : {duree:.1f} s")
    print(f"  Utilisateurs     : {n_traites}")
    print(f"  Segments totaux  : {total_s}")
    print(f"  Segments gardes  : {total_g}  "
          f"({100*total_g/total_s:.1f}%)" if total_s else
          "  Segments gardes  : 0")
    print(f"  Segments forts   : {total_f}")
    print(f"  Synthese         : {chemin_synthese}")
    print(f"  Detail gardes    : {sortie_dir}/<user>_H_gardes.csv")

    # Stats globales detaillees
    if toutes_lignes:
        afficher_stats_globales(toutes_lignes)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Filtre les <user>_H.csv et produit les H globaux "
                    "'gardes' et 'forts'.")
    p.add_argument("--source", default=SOURCE_DIR)
    p.add_argument("--sortie", default=SORTIE_DIR)
    p.add_argument("--seuil-divergence", type=float, default=SEUIL_DIVERGENCE)
    p.add_argument("--seuil-fort", type=float, default=SEUIL_FORT)
    args = p.parse_args()
    main(args.source, args.sortie, args.seuil_divergence, args.seuil_fort)