"""
analyse_tdrive.py
=================
Analyse exploratoire des fichiers T-Drive.

Pour chaque fichier taxi du dossier DOSSIER_TDRIVE :
  - Lit les points GPS bruts
  - Supprime les doublons stricts (meme timestamp)
  - Trie par temps
  - Calcule delta_t, distance et vitesse instantanee entre points consecutifs
  - UN FICHIER = UN SEGMENT (pas de coupure temporelle)
  - Signale dans le log les gaps > SEUIL_GAP_NOTABLE (informatif uniquement)
  - Produit un log .txt detaille par taxi (point par point, sans limitation)
  - Accumule les stats agregees dans un CSV global

REPRISE AUTOMATIQUE
-------------------
Si stats_tdrive.csv existe deja, les taxis deja traites sont
detectes via la colonne taxi_id et sautes. On peut donc
interrompre (Ctrl+C, panne, batterie, etc.) et relancer sans perte.

FORMAT T-DRIVE
--------------
Chaque ligne d'un fichier taxi : taxi_id,datetime,longitude,latitude
Exemple : 1,2008-02-02 13:34:05,116.51172,39.92123

FREQUENCE D'ECHANTILLONNAGE
----------------------------
Les auteurs annoncent un intervalle moyen de 177s avec une distance
moyenne de 623m. L'inspection du fichier 1.txt montre un pas dominant
de 10 minutes entre points consecutifs, avec des variations.
SEUIL_GAP_NOTABLE = 600s (10 minutes) : tout gap depassant ce seuil
est signale dans le log comme informatif. Pas de coupure de segment.
"""

import os
import csv
import math
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

DOSSIER_TDRIVE = "./dataset/T-drive Taxi Trajectories/release/taxi_log_2008_by_id"
DOSSIER_LOGS   = "./logs_tdrive"
CSV_STATS        = "./logs_tdrive/csv/stats_tdrive.csv"

# --- Seuil de signalement des gaps notables ---
# Un gap > SEUIL_GAP_NOTABLE est annote dans le log (informatif, pas de coupure).
# Regle sur la frequence annoncee par les auteurs (10 minutes observees dans 1.txt).
# Ajuster si necessaire apres inspection des premiers logs.
SEUIL_GAP_NOTABLE = 600   # secondes (10 minutes)


# ============================================================
# Lecture d'un fichier T-Drive
# ============================================================

def lire_fichier_tdrive(chemin):
    """
    Lit un fichier taxi T-Drive et retourne une liste de tuples :
    (taxi_id, timestamp, lon, lat, datetime_str)

    Format d'entree : taxi_id,datetime,longitude,latitude
    Exemple : 1,2008-02-02 13:34:05,116.51172,39.92123

    Les lignes malformees sont ignorees silencieusement.
    Filtre basique : coordonnees GPS plausibles pour Pekin
    (lat 38-42, lon 114-120).
    """
    points  = []
    taxi_id = os.path.splitext(os.path.basename(chemin))[0]

    with open(chemin, "r", encoding="utf-8", errors="replace") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue

            parties = ligne.split(",")
            if len(parties) < 4:
                continue

            try:
                dt_str = parties[1].strip()
                dt     = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                lon    = float(parties[2].strip())
                lat    = float(parties[3].strip())
                ts     = dt.timestamp()
            except (ValueError, IndexError):
                continue

            # Filtre coordonnees hors zone Pekin
            # if not (38.0 < lat < 42.0 and 114.0 < lon < 120.0):
            #     continue

            points.append((taxi_id, ts, lon, lat, dt_str))

    return points


# ============================================================
# Tri + deduplication
# ============================================================

def trier_et_dedupliquer(points):
    """
    Trie par timestamp puis supprime les doublons stricts
    (meme timestamp consecutif apres tri).
    Pas de filtre de vitesse, pas de coupure.

    Retourne (points_propres, nb_doublons).
    """
    tries = sorted(points, key=lambda p: p[1])

    propres = []
    for p in tries:
        if propres and p[1] == propres[-1][1]:
            continue
        propres.append(p)

    return propres, len(points) - len(propres)


# ============================================================
# Calculs geo
# ============================================================

def haversine(lon1, lat1, lon2, lat2):
    """
    Distance en metres entre deux points (lon, lat) en degres.
    Formule haversine, rayon terrestre R = 6 371 000 m.
    """
    R    = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = (math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def vitesse_kmh(distance_m, delta_t_s):
    """Vitesse en km/h. None si delta_t <= 0."""
    if delta_t_s <= 0:
        return None
    return (distance_m / 1000.0) / (delta_t_s / 3600.0)


# ============================================================
# Calcul des stats inter-points (un fichier = un segment)
# ============================================================

def calculer_inter_points(points):
    """
    Calcule delta_t, distance et vitesse instantanee entre chaque
    paire de points consecutifs. Un fichier entier = un seul segment.

    Retourne une liste de dict, un par point :
    {
        "point"      : (taxi_id, ts, lon, lat, datetime_str),
        "delta_t"    : float ou None   (None pour le premier point),
        "distance"   : float ou None,
        "vitesse"    : float ou None,
        "gap_notable": bool,  # delta_t > SEUIL_GAP_NOTABLE
    }
    """
    resultat = []

    for i, point in enumerate(points):
        if i == 0:
            resultat.append({
                "point"      : point,
                "delta_t"    : None,
                "distance"   : None,
                "vitesse"    : None,
                "gap_notable": False,
            })
            continue

        prev = points[i - 1]
        dt   = point[1] - prev[1]
        dist = haversine(prev[2], prev[3], point[2], point[3])
        v    = vitesse_kmh(dist, dt)

        resultat.append({
            "point"      : point,
            "delta_t"    : dt,
            "distance"   : dist,
            "vitesse"    : v,
            "gap_notable": dt > SEUIL_GAP_NOTABLE,
        })

    return resultat


# ============================================================
# Statistiques agregees
# ============================================================

def stats_agregees(entrees):
    """
    Calcule les statistiques agregees sur l'ensemble des points
    d'un taxi (= un segment dans notre cas).

    Ignore le premier point (pas de delta_t ni de distance).
    """
    n = len(entrees)

    delta_ts  = [e["delta_t"]  for e in entrees[1:] if e["delta_t"]  is not None]
    distances = [e["distance"] for e in entrees[1:] if e["distance"] is not None]
    vitesses  = [e["vitesse"]  for e in entrees[1:] if e["vitesse"]  is not None]

    def mediane(lst):
        if not lst:
            return None
        s = sorted(lst)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m-1] + s[m]) / 2

    def ecart_type(lst):
        if len(lst) < 2:
            return None
        mu = sum(lst) / len(lst)
        return math.sqrt(sum((x - mu) ** 2 for x in lst) / len(lst))

    duree = None
    if n >= 2:
        duree = entrees[-1]["point"][1] - entrees[0]["point"][1]

    return {
        "nb_points"        : n,
        "duree_s"          : round(duree, 1)           if duree      else None,
        "dt_median"        : round(mediane(delta_ts), 1) if delta_ts  else None,
        "dt_max"           : round(max(delta_ts), 1)   if delta_ts   else None,
        "nb_gaps_notables" : sum(1 for e in entrees[1:] if e["gap_notable"]),
        "dist_median"      : round(mediane(distances), 1) if distances else None,
        "dist_max"         : round(max(distances), 1)  if distances  else None,
        "dist_totale"      : round(sum(distances), 1)  if distances  else None,
        "v_median"         : round(mediane(vitesses), 2) if vitesses  else None,
        "v_max"            : round(max(vitesses), 2)   if vitesses   else None,
        "v_ecart_type"     : round(ecart_type(vitesses), 2) if vitesses else None,
    }


# ============================================================
# Ecriture du log .txt par taxi (sans limitation de points)
# ============================================================

def ecrire_log_taxi(taxi_id, chemin_log, entrees, nb_bruts, nb_doublons):
    """
    Ecrit le log .txt detaille pour un taxi.
    Tous les points sont affiches (pas de limitation).
    Les gaps > SEUIL_GAP_NOTABLE sont annotes.
    """
    os.makedirs(os.path.dirname(chemin_log), exist_ok=True)

    st = stats_agregees(entrees)

    with open(chemin_log, "w", encoding="utf-8") as f:

        # En-tete
        f.write(f"LOG TAXI {taxi_id}\n")
        f.write(f"Points bruts lus       : {nb_bruts}\n")
        f.write(f"Doublons supprimes     : {nb_doublons}\n")
        f.write(f"Points apres nettoyage : {len(entrees)}\n")
        f.write(f"SEUIL_GAP_NOTABLE      : {SEUIL_GAP_NOTABLE}s\n")
        f.write("\n")

        # Stats agregees
        f.write("--- Statistiques globales ---\n")
        f.write(f"Duree totale           : {st['duree_s']}s\n")
        f.write(f"Delta-t median         : {st['dt_median']}s\n")
        f.write(f"Delta-t max            : {st['dt_max']}s\n")
        f.write(f"Gaps notables (>{SEUIL_GAP_NOTABLE}s) : {st['nb_gaps_notables']}\n")
        f.write(f"Distance mediane       : {st['dist_median']}m\n")
        f.write(f"Distance max           : {st['dist_max']}m\n")
        f.write(f"Distance totale        : {st['dist_totale']}m\n")
        f.write(f"Vitesse mediane        : {st['v_median']} km/h\n")
        f.write(f"Vitesse max            : {st['v_max']} km/h\n")
        f.write(f"Ecart-type vitesse     : {st['v_ecart_type']} km/h\n")
        f.write("=" * 70 + "\n\n")

        # Detail point par point
        for idx, e in enumerate(entrees):
            _, ts, lon, lat, dt_str = e["point"]
            dt   = e["delta_t"]
            dist = e["distance"]
            v    = e["vitesse"]

            ligne = f"[{idx:6d}] {dt_str} | lon={lon:.6f} lat={lat:.6f}"
            if dt   is not None: ligne += f" | dt={dt:.0f}s"
            if dist is not None: ligne += f" | dist={dist:.1f}m"
            if v    is not None: ligne += f" | v={v:.1f}km/h"

            f.write(ligne + "\n")

            if e["gap_notable"]:
                f.write(f"          >> GAP NOTABLE : delta_t={dt:.0f}s"
                        f" > {SEUIL_GAP_NOTABLE}s"
                        f" | dist={dist:.1f}m"
                        f" | v_implicite={v:.1f}km/h\n")


# ============================================================
# CSV global des stats (avec reprise automatique)
# ============================================================

COLONNES_CSV = [
    "taxi_id",
    "nb_points", "duree_s",
    "dt_median", "dt_max", "nb_gaps_notables",
    "dist_median", "dist_max", "dist_totale",
    "v_median", "v_max", "v_ecart_type",
]


def charger_taxis_deja_traites(chemin_csv):
    """
    Lit le CSV global et retourne l'ensemble des taxi_id deja traites.
    Permet la reprise automatique apres interruption.
    """
    deja_traites = set()
    if not os.path.exists(chemin_csv):
        return deja_traites

    with open(chemin_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for ligne in reader:
            deja_traites.add(ligne["taxi_id"])

    return deja_traites


def ecrire_stats_csv(chemin_csv, taxi_id, st):
    """
    Ajoute une ligne de stats au CSV global.
    Cree le fichier avec en-tete si necessaire.
    Mode append : ne jamais ecraser les resultats precedents.
    """
    fichier_existe = os.path.exists(chemin_csv)

    with open(chemin_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES_CSV)

        if not fichier_existe:
            writer.writeheader()

        writer.writerow({
            "taxi_id"          : taxi_id,
            "nb_points"        : st["nb_points"],
            "duree_s"          : st["duree_s"],
            "dt_median"        : st["dt_median"],
            "dt_max"           : st["dt_max"],
            "nb_gaps_notables" : st["nb_gaps_notables"],
            "dist_median"      : st["dist_median"],
            "dist_max"         : st["dist_max"],
            "dist_totale"      : st["dist_totale"],
            "v_median"         : st["v_median"],
            "v_max"            : st["v_max"],
            "v_ecart_type"     : st["v_ecart_type"],
        })


# ============================================================
# Main
# ============================================================

def main():

    if not os.path.isdir(DOSSIER_TDRIVE):
        print(f"[ERREUR] Dossier introuvable : {DOSSIER_TDRIVE}")
        return

    fichiers = sorted([
        f for f in os.listdir(DOSSIER_TDRIVE)
        if f.endswith(".txt")
    ])

    if not fichiers:
        print(f"[ERREUR] Aucun fichier .txt dans {DOSSIER_TDRIVE}")
        return

    print(f"Dataset T-Drive : {len(fichiers)} fichiers")
    print(f"SEUIL_GAP_NOTABLE = {SEUIL_GAP_NOTABLE}s")
    print(f"Un fichier = un segment (pas de coupure temporelle)")
    print(f"CSV stats  : {CSV_STATS}")
    print(f"Logs texte : {DOSSIER_LOGS}/")
    print()

    # Reprise automatique
    deja_traites = charger_taxis_deja_traites(CSV_STATS)
    if deja_traites:
        print(f"Reprise : {len(deja_traites)} taxi(s) deja traites — ignores.")
        print()

    nb_traites = 0
    nb_ignores = 0
    nb_erreurs = 0

    for nom_fichier in fichiers:
        taxi_id = os.path.splitext(nom_fichier)[0]
        chemin  = os.path.join(DOSSIER_TDRIVE, nom_fichier)

        if taxi_id in deja_traites:
            nb_ignores += 1
            continue

        try:
            points_bruts             = lire_fichier_tdrive(chemin)
            nb_bruts                 = len(points_bruts)

            if nb_bruts == 0:
                print(f"  {taxi_id} : vide ou aucun point valide — ignore")
                nb_erreurs += 1
                continue

            points_propres, nb_doublons = trier_et_dedupliquer(points_bruts)
            entrees                     = calculer_inter_points(points_propres)
            st                          = stats_agregees(entrees)

            chemin_log = os.path.join(DOSSIER_LOGS, f"log_{taxi_id}.txt")
            ecrire_log_taxi(taxi_id, chemin_log, entrees, nb_bruts, nb_doublons)
            ecrire_stats_csv(CSV_STATS, taxi_id, st)

            print(f"  {taxi_id:6s} : {nb_bruts:6d} pts bruts"
                  f" | {nb_doublons:4d} doublons"
                  f" | {len(entrees):6d} pts"
                  f" | dt_med={st['dt_median']}s"
                  f" | v_max={st['v_max']}km/h"
                  f" | gaps={st['nb_gaps_notables']}")

            nb_traites += 1

        except KeyboardInterrupt:
            print("\n[INTERRUPTION] Ctrl+C detecte.")
            print(f"Progression sauvegardee dans {CSV_STATS}")
            print(f"Relancer le script pour reprendre.")
            break

        except Exception as e:
            print(f"  {taxi_id} : ERREUR — {e}")
            nb_erreurs += 1
            continue

    print()
    print(f"Termine.")
    print(f"  Traites cette execution : {nb_traites}")
    print(f"  Ignores (deja traites)  : {nb_ignores}")
    print(f"  Erreurs                 : {nb_erreurs}")


if __name__ == "__main__":
    main()
