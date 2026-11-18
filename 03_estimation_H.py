#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_estimation_H.py  (ex estimation_H_all_seg.py)

Lance l'estimation de H sur TOUS les fichiers CSV présents dans ./sortie/.
Avec reprise automatique, gestion de Ctrl+C, et timeout global (10 min).

Ce qui est identique à la V4 : les 13 premières colonnes de resultats_H/<user>_H.csv,
les affichages et les logs (log_estimation_H/), la reprise, le Ctrl+C, le timeout.

Ce qui est NOUVEAU (colonnes ajoutées à la fin de chaque ligne, rien n'est retiré) :
  - versions rapides (numpy) des trois méthodes, identiques aux versions V4 à 1e-9 près
  - h_rs_corrige : R/S corrigé du biais de petit échantillon (Anis-Lloyd)
  - dfa_bas, dfa_haut, ecart_pentes : pentes DFA des moitiés basse/haute des échelles
  - dfa_sup_1 : exposant DFA > 1 (hors du domaine du bruit gaussien fractionnaire)
  - whittle_m, whittle_a_la_borne : Whittle collé à la borne de la grille = non fiable
  - nul_p, nul_seuil, nul_exces, nul_ordre_ar : test de dépassement d'un modèle à mémoire courte AR(p)
  - mode_echantillonnage, pas_median_s, cv_dt, corr_d_dt : le logger plutôt que le déplacement ?
  - h_dfa_morceaux, n_morceaux : H après coupure aux artefacts, SANS raccorder les morceaux
    (remplace le défaut de "H sans suspects", qui raccordait les voisins)
  - empreinte_config : empreinte de config.py au moment du calcul

Options :  --sortie DOSSIER  --resultats DOSSIER  --logs DOSSIER
           --sans-nul (plus rapide)  --serie distance|vitesse  --ordre-ar 1|5
"""

import os
import sys
import csv
from datetime import datetime

import config
from hurst.dfa import dfa_h
from hurst.estimation_segment import estimer_segment, series_du_segment, h_dfa_par_morceaux
from hurst.statistiques_segment import statistiques_segment, afficher_statistiques


# ============================================================
# CONSTANTES
# ============================================================

MIN_POINTS_H_SANS_SUSPECTS = 20
SEUIL_ECART_H_SUSPECT = 0.10
SORTIE_DIR = "./sortie"
RESULTATS_DIR = "resultats_H"
LOG_DIR = "log_estimation_H"
TIMEOUT_SECONDS = 600  # 10 minutes

# Reglages V5 (valeurs par defaut dans config.py ; modifiables en ligne de commande)
SERIE = config.SERIE_ANALYSEE         # "distance" (d_i) ou "vitesse" (v_i)
AVEC_NUL = config.AVEC_TEST_NUL       # test de depassement d'un modele a memoire courte
ORDRE_AR = config.NUL_ORDRE_AR_MAX    # ordre maximal du modele nul AR(p) : presenter p = 1 ET p = 5

# Colonnes ajoutees par la V5, apres les 13 colonnes de la V4 (dans cet ordre)
COLONNES_V5 = ["h_rs_corrige", "dfa_bas", "dfa_haut", "ecart_pentes", "dfa_sup_1", "whittle_m",
               "whittle_a_la_borne", "nul_p", "nul_seuil", "nul_exces", "nul_ordre_ar", "mode_echantillonnage",
               "pas_median_s", "cv_dt", "corr_d_dt", "cv_d", "part_immobile", "rapport_net",
               "duree_s", "trajet_m", "serie", "empreinte_config"]


def _ou_none(x):
    """None si la valeur est absente ou NaN (convention de la V4 : None = non calculable)."""
    if x is None:
        return None
    try:
        return None if x != x else float(x)
    except (TypeError, ValueError):
        return x


def _valeur_csv(x):
    """Valeur pour le CSV : NaN -> vide ; booleens numpy -> bool ; le reste inchange."""
    if x is None:
        return None
    if isinstance(x, float) and x != x:
        return None
    if hasattr(x, "item"):
        return x.item()
    return x


# ============================================================
# FONCTIONS DE L'INFO 1/3
# ============================================================

def lire_fichier_csv(file_path):
    points = []
    try:
        with open(file_path, "r", newline="") as f:
            lecteur = csv.DictReader(f)
            for num_ligne, ligne in enumerate(lecteur, start=2):
                try:
                    segment_id = int(float(ligne["segment_id"]))
                    lat = float(ligne["lat"])
                    lon = float(ligne["lon"])
                    timestamp = float(ligne["timestamp"])
                    suspect_dist = ligne.get("suspect_dist", "False").strip().lower() == "true"
                    suspect_vitesse = ligne.get("suspect_vitesse", "False").strip().lower() == "true"
                    type_suspect = (ligne.get("type_suspect_vitesse") or "").strip()
                except (ValueError, KeyError, AttributeError) as e:
                    print(f"[AVERTISSEMENT] Ligne {num_ligne} non parseable → {e}")
                    continue
                points.append({
                    "segment_id": segment_id,
                    "lat": lat,
                    "lon": lon,
                    "timestamp": timestamp,
                    "suspect_dist": suspect_dist,
                    "suspect_vitesse": suspect_vitesse,
                    "type_suspect_vitesse": type_suspect,
                })
    except OSError as e:
        print(f"[ERREUR] Impossible d'ouvrir {file_path} → {e}")
    return points


def reconstruire_segments(points):
    """
    Regroupe les points lus par segment_id.

    Sortie : liste de dicts, un par segment :
        segment_id      : identifiant dans le CSV de sortie
        points_complets : [[lat, lon, timestamp], ...] -- TOUS les points (V4)
        points_filtres  : idem sans les points marques suspect_dist / suspect_vitesse (V4)
        df_marque       : DataFrame (lat, lon, timestamp, suspect_dist, type_suspect_vitesse)
                          pour la coupure aux artefacts (V5)
    """
    import pandas as pd

    segments = []
    segment_complet = []
    segment_filtre = []
    marques = []
    seg_id_courant = None

    def _clore():
        df = pd.DataFrame(
            [[pt[0], pt[1], pt[2], md, ts] for pt, (md, ts) in zip(segment_complet, marques)],
            columns=["lat", "lon", "timestamp", "suspect_dist", "type_suspect_vitesse"])
        segments.append({
            "segment_id": seg_id_courant,
            "points_complets": segment_complet,
            "points_filtres": segment_filtre,
            "df_marque": df,
        })

    for p in points:
        if p["segment_id"] != seg_id_courant:
            if segment_complet:
                _clore()
            segment_complet = []
            segment_filtre = []
            marques = []
            seg_id_courant = p["segment_id"]
        pt = [p["lat"], p["lon"], p["timestamp"]]
        segment_complet.append(pt)
        marques.append((p["suspect_dist"], p.get("type_suspect_vitesse", "")))
        if not (p["suspect_dist"] or p["suspect_vitesse"]):
            segment_filtre.append(pt)

    if segment_complet:
        _clore()

    return segments


def sauvegarder_resultats_H(user_id, lignes, dossier=RESULTATS_DIR):
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, f"{user_id}_H.csv")
    with open(chemin, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_id", "n_points", "h_dfa", "h_rs", "h_whittle",
            "distance_max_m", "vitesse_max_kmh", "delta_t_moyen_s",
            "divergence", "n_points_suspects",
            "h_dfa_sans_suspects", "ecart_h_dfa_suspects", "suspect"
        ] + COLONNES_V5 + ["h_dfa_morceaux", "n_morceaux"])
        writer.writerows(lignes)
    return chemin


# ============================================================
# FONCTIONS DE L'INFO 2/3
# ============================================================

def calculer_divergence(h_dfa, h_rs, h_whittle):
    valeurs = [h for h in [h_dfa, h_rs, h_whittle] if h is not None]
    if len(valeurs) < 2:
        return None
    return round(max(valeurs) - min(valeurs), 7)


def check_dfa(h_dfa):
    if h_dfa is None:
        return 'OK', 1
    if h_dfa > 1:
        return 'DFA', h_dfa
    return 'OK', 1


def check_rs(h_rs):
    if h_rs is None:
        return 'OK', 1
    if h_rs > 1:
        return 'RS', h_rs
    return 'OK', 1


def check_wh(h_wh):
    if h_wh is None:
        return 'OK', 1
    if h_wh > 0.999:
        return 'WH', h_wh
    return 'OK', 1


# ============================================================
# CLASSE TEE (redirection stdout)
# ============================================================

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# ============================================================
# TRAITEMENT D'UN UTILISATEUR
# ============================================================

def traiter_un_utilisateur(user_id, csv_path):
    points = lire_fichier_csv(csv_path)
    if not points:
        print(f"[AVERTISSEMENT] Aucun point pour {user_id}.")
        return 0, 0, 0, None, None, None, None

    segments = reconstruire_segments(points)
    if not segments:
        print(f"[AVERTISSEMENT] Aucun segment pour {user_id}.")
        return 0, 0, 0, None, None, None, None

    lignes_csv = []
    suspects = 0
    ok = 0

    h_dfa_list = []
    h_rs_list = []
    h_whittle_list = []
    poids = []

    print(f"\n{'='*60}")
    print(f"Utilisateur {user_id}  —  {len(segments)} segments")
    print(f"{'='*60}")

    for j, seg in enumerate(segments):
        points_complets = seg["points_complets"]
        points_filtres = seg["points_filtres"]
        nb_suspects = len(points_complets) - len(points_filtres)

        # Versions rapides (V5) : memes valeurs que DFA(), RS(), Whittle() de la V4
        try:
            res = estimer_segment(points_complets, user_id, j, serie=SERIE, avec_nul=AVEC_NUL, ordre_ar_max=ORDRE_AR)
        except Exception as e:
            print(f"[AVERTISSEMENT] Segment {j} : estimation impossible → {e}")
            res = {}
        H_dfa = _ou_none(res.get("h_dfa"))
        H_rs = _ou_none(res.get("h_rs_brut"))
        H_whittle = _ou_none(res.get("h_whittle"))

        stats = statistiques_segment(points_complets) if points_complets else {}
        divergence = calculer_divergence(H_dfa, H_rs, H_whittle)
        n_points = stats.get("nb_points", 0)

        print(f"\nSegment {j} : H (DFA) = {H_dfa}")
        print(f"Segment {j} : H (RS)  = {H_rs}")
        print(f"Segment {j} : H (WHITTLE) = {H_whittle}")
        afficher_statistiques(j, H_dfa, stats)

        msg_dfa, val_dfa = check_dfa(H_dfa)
        msg_rs, val_rs = check_rs(H_rs)
        msg_wh, val_wh = check_wh(H_whittle)

        suspect_flag = False
        if divergence is not None:
            if divergence > 0.15 or msg_dfa != 'OK' or msg_rs != 'OK' or msg_wh != 'OK':
                suspect_flag = True
                suspects += 1
                flag = " ⚠ SUSPECT"
                if msg_dfa != 'OK':
                    print(f'{msg_dfa} : {val_dfa}')
                if msg_rs != 'OK':
                    print(f'{msg_rs} : {val_rs}')
                if msg_wh != 'OK':
                    print(f'{msg_wh} : {val_wh}')
            else:
                ok += 1
                flag = " ✓"
            print(f"Segment {j} : divergence inter-méthodes = {divergence}{flag}")

        H_dfa_sans_suspects = None
        ecart_H_dfa_suspects = None

        if nb_suspects > 0:
            if len(points_filtres) >= MIN_POINTS_H_SANS_SUSPECTS:
                try:
                    H_dfa_sans_suspects = _ou_none(dfa_h(series_du_segment(points_filtres)["d"]))
                except Exception:
                    H_dfa_sans_suspects = None
                if H_dfa_sans_suspects is not None and H_dfa is not None:
                    ecart_H_dfa_suspects = round(abs(H_dfa - H_dfa_sans_suspects), 7)

                print(f"Segment {j} : {nb_suspects} point(s) suspect(s) "
                      f"({len(points_complets)} -> {len(points_filtres)} points)")
                print(f"Segment {j} : H (DFA) sans points suspects = {H_dfa_sans_suspects}")
                if ecart_H_dfa_suspects is not None:
                    flag_ecart = " ⚠ DEPENDANT DES POINTS SUSPECTS" if ecart_H_dfa_suspects > SEUIL_ECART_H_SUSPECT else " ✓"
                    print(f"Segment {j} : écart H (avec/sans suspects) = {ecart_H_dfa_suspects}{flag_ecart}")
            else:
                print(f"Segment {j} : {nb_suspects} point(s) suspect(s), mais "
                      f"trop peu de points restants ({len(points_filtres)})")

        # H apres COUPURE aux artefacts, sans raccord (V5) -- remplace le raccord de "sans suspects"
        try:
            H_dfa_morceaux, nb_morceaux, _ = h_dfa_par_morceaux(seg["df_marque"], MIN_POINTS_H_SANS_SUSPECTS)
        except Exception:
            H_dfa_morceaux, nb_morceaux = None, None
        if nb_suspects > 0 and H_dfa_morceaux is not None:
            print(f"Segment {j} : H (DFA) apres coupure aux artefacts (sans raccord) = "
                  f"{round(H_dfa_morceaux, 7)}  ({nb_morceaux} morceau(x))")

        if H_dfa is not None:
            h_dfa_list.append(H_dfa)
            poids.append(n_points)
        if H_rs is not None:
            h_rs_list.append(H_rs)
        if H_whittle is not None:
            h_whittle_list.append(H_whittle)

        lignes_csv.append([
            j,
            n_points,
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
            1 if suspect_flag else 0,
            *[_valeur_csv(res.get(c)) for c in COLONNES_V5],
            _valeur_csv(H_dfa_morceaux),
            nb_morceaux,
        ])

    # H global
    h_global_dfa_avg = round(sum(h_dfa_list) / len(h_dfa_list), 6) if h_dfa_list else None
    h_global_rs_avg = round(sum(h_rs_list) / len(h_rs_list), 6) if h_rs_list else None
    h_global_wh_avg = round(sum(h_whittle_list) / len(h_whittle_list), 6) if h_whittle_list else None

    if h_dfa_list and poids:
        h_global_dfa_wavg = round(sum(h * p for h, p in zip(h_dfa_list, poids)) / sum(poids), 6)
    else:
        h_global_dfa_wavg = None

    print(f"\n{'-'*60}")
    print(f"Résumé utilisateur {user_id} :")
    print(f"  SUSPECT : {suspects}/{len(segments)} segments")
    print(f"  OK      : {ok}/{len(segments)} segments")
    print(f"  H global (DFA) moyenne arithmétique : {h_global_dfa_avg}")
    print(f"  H global (DFA) moyenne pondérée     : {h_global_dfa_wavg}")
    print(f"{'-'*60}")

    chemin_csv = sauvegarder_resultats_H(user_id, lignes_csv, RESULTATS_DIR)

    return len(segments), suspects, ok, h_global_dfa_avg, h_global_rs_avg, h_global_wh_avg, chemin_csv


# ============================================================
# MAIN AVEC GESTION DE Ctrl+C ET TIMEOUT
# ============================================================

def main():
    if not os.path.exists(SORTIE_DIR):
        print(f"[ERREUR] Dossier {SORTIE_DIR} introuvable.")
        return

    os.makedirs(RESULTATS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(SORTIE_DIR) if f.endswith('.csv')])
    print(f"Fichiers CSV trouvés : {len(csv_files)}")

    # Utilisateurs déjà traités
    deja_traites = set()
    for f in os.listdir(RESULTATS_DIR):
        if f.endswith('_H.csv'):
            user = f.replace('_H.csv', '')
            deja_traites.add(user)
    print(f"Utilisateurs déjà traités : {len(deja_traites)}")

    total_segments = 0
    total_suspects = 0
    total_ok = 0
    log_file = None

    # Heure de début du script
    script_start = datetime.now()
    print(f"⏱️  Temps limite : {TIMEOUT_SECONDS} secondes ({TIMEOUT_SECONDS//60} minutes)")
    print(f"🚀 Début du script : {script_start.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        for csv_file in csv_files:
            # Vérification du timeout avant de commencer un nouvel utilisateur
            elapsed = (datetime.now() - script_start).total_seconds()
            if elapsed > TIMEOUT_SECONDS:
                print(f"\n⏱️  Temps limite atteint ({elapsed:.0f}s > {TIMEOUT_SECONDS}s).")
                print("Arrêt programmé après le dernier utilisateur traité.")
                break

            user_id = csv_file.replace('.csv', '')
            csv_path = os.path.join(SORTIE_DIR, csv_file)

            # Reprise
            resultat_path = os.path.join(RESULTATS_DIR, f"{user_id}_H.csv")
            if os.path.exists(resultat_path) and os.path.getsize(resultat_path) > 0:
                print(f"[INFO] Utilisateur {user_id} déjà traité, ignoré.")
                continue

            # Redirection stdout
            log_path = os.path.join(LOG_DIR, f"log_estimation_{user_id}.txt")
            log_file = open(log_path, "w", encoding="utf-8")
            original_stdout = sys.stdout
            sys.stdout = Tee(original_stdout, log_file)

            start_time = datetime.now()
            print(f"\n{'#'*60}")
            print(f"# Début du traitement pour l'utilisateur {user_id}")
            print(f"# Heure de début : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"# Temps écoulé depuis lancement : {elapsed:.0f}s / {TIMEOUT_SECONDS}s")
            print(f"{'#'*60}")

            nb_seg, nb_susp, nb_ok, h_avg, h_rs_avg, h_wh_avg, chemin_resultat = traiter_un_utilisateur(user_id, csv_path)

            end_time = datetime.now()
            duration = end_time - start_time
            elapsed_after = (datetime.now() - script_start).total_seconds()

            print(f"\n{'#'*60}")
            print(f"# Fin du traitement pour l'utilisateur {user_id}")
            print(f"# Durée : {duration.total_seconds():.2f} secondes")
            print(f"# Temps écoulé depuis lancement : {elapsed_after:.0f}s / {TIMEOUT_SECONDS}s")
            print(f"# Résultats CSV : {chemin_resultat}")
            print(f"# Logs : {log_path}")
            print(f"# H global (DFA) : {h_avg}")
            print(f"{'#'*60}")

            sys.stdout = original_stdout
            log_file.close()
            log_file = None

            total_segments += nb_seg
            total_suspects += nb_susp
            total_ok += nb_ok

            print(f"[INFO] Utilisateur {user_id} traité. Logs dans {log_path}")

    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("⏹️  Interruption clavier détectée (Ctrl+C).")
        print("Arrêt propre du pipeline en cours...")
        if log_file is not None:
            log_file.close()
            sys.stdout = original_stdout
            print("[INFO] Fichier de log fermé.")
        print(f"  Utilisateurs traités dans cette session : {len(csv_files) - len(deja_traites) - (1 if log_file else 0)}")
        print(f"  Total segments sauvegardés : {total_segments}")
        print(f"  Total SUSPECT : {total_suspects}")
        print(f"  Total OK      : {total_ok}")
        print("="*60)
        print("✅ Données déjà sauvegardées. Relancez le script pour continuer.")
        return

    # Bilan final
    elapsed_final = (datetime.now() - script_start).total_seconds()
    print(f"\n{'='*60}")
    print("BILAN GLOBAL DE L'EXÉCUTION")
    print(f"{'='*60}")
    print(f"  Temps total écoulé : {elapsed_final:.0f} secondes ({elapsed_final/60:.1f} minutes)")
    print(f"  Utilisateurs traités : {len(csv_files) - len(deja_traites)} (nouveaux)")
    print(f"  Utilisateurs ignorés  : {len(deja_traites)} (déjà traités)")
    print(f"  Total segments        : {total_segments}")
    print(f"  Total SUSPECT         : {total_suspects}")
    print(f"  Total OK              : {total_ok}")
    print(f"  Logs dans             : {LOG_DIR}/")
    print(f"  Résultats CSV dans    : {RESULTATS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    analyseur = argparse.ArgumentParser(description="Estimation de H sur tous les CSV de sortie/")
    analyseur.add_argument("--sortie", default=SORTIE_DIR, help="dossier des CSV de pretraitement")
    analyseur.add_argument("--resultats", default=RESULTATS_DIR, help="dossier des resultats <user>_H.csv")
    analyseur.add_argument("--logs", default=LOG_DIR, help="dossier des logs d'estimation")
    analyseur.add_argument("--sans-nul", action="store_true", help="ne pas calculer le test nul (plus rapide)")
    analyseur.add_argument("--serie", choices=["distance", "vitesse"], default=SERIE)
    analyseur.add_argument("--ordre-ar", type=int, default=ORDRE_AR, help="ordre maximal du modele nul AR(p) (1 ou 5)")
    arguments = analyseur.parse_args()
    SORTIE_DIR, RESULTATS_DIR, LOG_DIR = arguments.sortie, arguments.resultats, arguments.logs
    AVEC_NUL = AVEC_NUL and not arguments.sans_nul
    SERIE = arguments.serie
    ORDRE_AR = arguments.ordre_ar
    main()