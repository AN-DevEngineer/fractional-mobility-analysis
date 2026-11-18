# convert_csv_to_plt.py
#
# Convertit un CSV exporté par l'app Flutter au format .plt GeoLife.
#
# Entrée : CSV avec colonnes : latitude, longitude, date, time, accuracy
# Sortie : fichier .plt avec en-tête GeoLife + données converties
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # racine du projet : permet `python outils/<script>.py`

import os
import csv
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# --- Fichiers ---
CSV_SOURCE = "./gps_all_20260803_1252.csv"   # Ton CSV de l'app
PLT_DEST   = "./mes_donnees.plt"                 # Fichier .plt de sortie

# --- Optionnel : ne convertir qu'une partie du CSV ---
# Si ces variables sont None, tout le fichier est converti.
DEBUT_LIGNES = None   # ex: 10 pour commencer à la ligne 10
FIN_LIGNES   = None   # ex: 200 pour s'arrêter à la ligne 200


# ============================================================
# Conversion
# ============================================================

def date_to_excel_serial(dt):
    """
    Convertit un datetime en nombre de jours depuis le 30/12/1899
    (format Excel / GeoLife).
    """
    excel_epoch = datetime(1899, 12, 30, 0, 0, 0)
    delta = dt - excel_epoch
    return delta.days + delta.seconds / 86400.0


def convertir_csv_vers_plt(csv_path, plt_path, debut=None, fin=None):
    """
    Lit le CSV de l'app et écrit un fichier .plt GeoLife.
    """
    if not os.path.exists(csv_path):
        print(f"[ERREUR] CSV introuvable : {csv_path}")
        return False

    # --- Lecture du CSV ---
    points = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        # Vérifier les colonnes
        if "latitude" not in reader.fieldnames:
            print("[ERREUR] Le CSV ne contient pas la colonne 'latitude'")
            return False

        for i, ligne in enumerate(reader):
            # Optionnel : ignorer les lignes avant debut / après fin
            if debut is not None and i < debut:
                continue
            if fin is not None and i > fin:
                break

            try:
                lat = float(ligne["latitude"])
                lon = float(ligne["longitude"])
                date_str = ligne["date"]
                time_str = ligne["time"]

                # Convertir en datetime
                dt = datetime.strptime(date_str + " " + time_str,
                                       "%Y-%m-%d %H:%M:%S")

                # Format Excel (jours depuis 30/12/1899)
                excel_date = date_to_excel_serial(dt)

                # Altitude : on met 0 (ou -777 si tu veux marquer "inconnue")
                altitude_ft = 0

                points.append((lat, lon, altitude_ft, excel_date, date_str, time_str))

            except (ValueError, KeyError) as e:
                print(f"Ligne {i} ignorée (format invalide) : {ligne} — {e}")
                continue

    if len(points) == 0:
        print("[ERREUR] Aucun point valide trouvé dans le CSV")
        return False

    # --- Écriture du fichier .plt ---
    with open(plt_path, "w") as f:
        # En-tête GeoLife (fixe)
        f.write("Geolife trajectory\n")
        f.write("WGS 84\n")
        f.write("Altitude is in Feet\n")
        f.write("Reserved 3\n")
        f.write("0,2,255,My Track,0,0,2,8421376\n")
        f.write("0\n")

        # Points
        for lat, lon, alt, excel_date, date_str, time_str in points:
            f.write(f"{lat:.6f},{lon:.6f},0,{alt:.0f},{excel_date:.12f},{date_str},{time_str}\n")

    print(f"[OK] Conversion terminée : {len(points)} points écrits dans {plt_path}")
    return True


def main():
    print("=" * 50)
    print("Conversion CSV → .plt (format GeoLife)")
    print("=" * 50)
    print(f"CSV source  : {CSV_SOURCE}")
    print(f"PLT cible   : {PLT_DEST}")
    print(f"Début lignes : {DEBUT_LIGNES if DEBUT_LIGNES is not None else 'début'}")
    print(f"Fin lignes   : {FIN_LIGNES if FIN_LIGNES is not None else 'fin'}")
    print()

    ok = convertir_csv_vers_plt(CSV_SOURCE, PLT_DEST, DEBUT_LIGNES, FIN_LIGNES)

    if ok:
        print("\nTu peux maintenant utiliser ce fichier .plt avec le script d'inspection.")
        print(f"  → Modifie FICHIER_PLT = \"{PLT_DEST}\" dans simuler_pretraitement.py")
    else:
        print("\n[ERREUR] La conversion a échoué.")


if __name__ == "__main__":
    main()