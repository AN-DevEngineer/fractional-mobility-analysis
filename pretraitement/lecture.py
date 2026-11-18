# ./pretraitement/lecture.py

import os
import calendar
from datetime import datetime


def convertir_timestamp(date_str, time_str):
    """
    Convertit une date et une heure GeoLife en timestamp UNIX (secondes), en UTC.

    Les horodatages GeoLife sont en GMT : on les lit comme tels. (La V4 utilisait
    datetime.timestamp(), donc l'heure locale de la machine : resultat dependant du
    fuseau horaire en cas de changement d'heure. Les ecarts de temps etaient justes,
    seul le timestamp absolu changeait.)

    Entree :
        date_str : "2008-10-23"
        time_str : "02:27:50"

    Sortie :
        float : timestamp en secondes (UTC)
    """
    dt = datetime.strptime(date_str + " " + time_str, "%Y-%m-%d %H:%M:%S")
    return float(calendar.timegm(dt.timetuple()))


def lire_fichier_plt(file_path):
    """
    Lit un fichier .plt GeoLife et retourne une liste de points

    Format .plt (après 6 lignes de header) :
        latitude, longitude, 0, altitude, date_float, date_str, time_str

    Entrée :
        file_path (str) : chemin vers le fichier .plt

    Sortie :
        list of tuples : [(lat, lon, alt, timestamp), ...]
        Retourne [] si le fichier est illisible
    """
    points = []

    try:
        with open(file_path, "r") as f:
            lignes = f.readlines()
    except OSError as e:
        print(f"[ERREUR] Impossible d'ouvrir : {file_path} → {e}")
        return points

    # Les 6 premières lignes sont un header fixe GeoLife
    for numero_ligne, ligne in enumerate(lignes[6:], start=7):

        ligne = ligne.strip()

        if ligne == "":
            continue

        parties = ligne.split(",")

        if len(parties) < 7:
            print(f"[AVERTISSEMENT] Ligne {numero_ligne} malformée dans {file_path}")
            continue

        try:
            lat      = float(parties[0])
            lon      = float(parties[1])
            alt      = float(parties[3])
            date_str = parties[5]
            time_str = parties[6]

            timestamp = convertir_timestamp(date_str, time_str)

            point = (lat, lon, alt, timestamp)
            points.append(point)

        except ValueError as e:
            print(f"[AVERTISSEMENT] Ligne {numero_ligne} non parseable dans {file_path} → {e}")
            continue

    return points


def lire_dossier_utilisateur(user_path):
    """
    Lit tous les fichiers .plt d'un utilisateur GeoLife

    Structure attendue :
        user_path/
            Trajectory/
                fichier1.plt
                fichier2.plt
                ...

    Entrée :
        user_path (str) : chemin vers le dossier de l'utilisateur

    Sortie :
        list of (str, list of tuples) :
            [ (chemin_fichier1, [(lat,lon,alt,t), ...]), ... ]
        Chaque élément correspond à un fichier .plt. Le chemin est
        conservé (et non plus seulement les points) afin de pouvoir
        l'indiquer comme "Fichier source" dans les logs de segments
        (cf. pretraitement/journal.py).
    """
    trajets = []

    dossier_traj = os.path.join(user_path, "Trajectory")

    if not os.path.exists(dossier_traj):
        print(f"[ERREUR] Dossier Trajectory introuvable : {dossier_traj}")
        return trajets

    for nom_fichier in sorted(os.listdir(dossier_traj)):  # sorted → ordre reproductible

        if not nom_fichier.endswith(".plt"):
            continue

        chemin = os.path.join(dossier_traj, nom_fichier)
        points = lire_fichier_plt(chemin)

        if len(points) > 0:
            trajets.append((chemin, points))

    return trajets
