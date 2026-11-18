#!/usr/bin/env python3
"""
carte_trajectoire.py

Trace les trajectoires GPS d'un fichier .plt sur une carte interactive
(OpenStreetMap via Folium). Produit un fichier HTML à ouvrir dans un
navigateur -- aucun serveur requis, fonctionne hors ligne une fois
les tuiles chargées.

Usage :
    pip install folium
    python3 carte_trajectoire.py mes_donnees.plt

Ou pour plusieurs fichiers (plusieurs jours) :
    python3 carte_trajectoire.py jour1.plt jour2.plt
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # racine du projet : permet `python outils/<script>.py`

import sys
import math
import argparse
from datetime import datetime

import folium
from folium.plugins import AntPath


# Couleurs pour distinguer les segments (en cas de segmentation a 600s)
COULEURS = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad",
            "#f39c12", "#16a085", "#c0392b", "#2c3e50"]

SEUIL_GAP_S = 600   # secondes -- meme valeur que dans le pipeline


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2*R*math.asin(math.sqrt(a))


def lire_plt(chemin):
    """
    Lit un fichier .plt (format GeoLife) et retourne une liste de
    dict {lat, lon, timestamp, datetime_str}.
    Ignore les 6 lignes d'en-tete.
    """
    points = []
    with open(chemin, encoding="utf-8", errors="replace") as f:
        lignes = f.readlines()

    for ligne in lignes[6:]:
        ligne = ligne.strip()
        if not ligne:
            continue
        parties = ligne.split(",")
        if len(parties) < 7:
            continue
        try:
            lat   = float(parties[0])
            lon   = float(parties[1])
            date  = parties[5].strip()
            heure = parties[6].strip()
            dt    = datetime.strptime(f"{date} {heure}", "%Y-%m-%d %H:%M:%S")
            points.append({"lat": lat, "lon": lon, "dt": dt,
                            "label": dt.strftime("%Y-%m-%d %H:%M:%S")})
        except (ValueError, IndexError):
            continue

    return points


def segmenter(points, seuil_s=SEUIL_GAP_S):
    """Decoupe en segments selon un seuil de gap temporel."""
    if not points:
        return []
    segments = []
    seg = [points[0]]
    for i in range(1, len(points)):
        dt = (points[i]["dt"] - points[i-1]["dt"]).total_seconds()
        if dt > seuil_s:
            segments.append(seg)
            seg = []
        seg.append(points[i])
    if seg:
        segments.append(seg)
    return segments


def stats_segment(seg):
    """Calcule quelques statistiques simples pour l'infobulle."""
    n = len(seg)
    duree = (seg[-1]["dt"] - seg[0]["dt"]).total_seconds() / 60
    dist = sum(
        haversine_m(seg[i-1]["lat"], seg[i-1]["lon"],
                    seg[i]["lat"],   seg[i]["lon"])
        for i in range(1, n)
    )
    return {"n": n, "duree_min": round(duree, 1), "dist_km": round(dist/1000, 2)}


def construire_carte(fichiers):
    """
    Construit la carte Folium à partir d'une liste de fichiers .plt.
    Chaque fichier est découpé en segments ; chaque segment a sa propre
    couleur et ses propres marqueurs de début/fin.
    """
    tous_points = []
    for chemin in fichiers:
        pts = lire_plt(chemin)
        tous_points.extend(pts)

    if not tous_points:
        print("[ERREUR] Aucun point valide trouvé.")
        return None

    # Centre de la carte : moyenne des positions
    lat_moy = sum(p["lat"] for p in tous_points) / len(tous_points)
    lon_moy = sum(p["lon"] for p in tous_points) / len(tous_points)

    carte = folium.Map(location=[lat_moy, lon_moy], zoom_start=14,
                       tiles="OpenStreetMap")

    couleur_idx = 0
    for chemin in fichiers:
        pts = lire_plt(chemin)
        if not pts:
            continue

        segments = segmenter(pts)
        nom_fichier = chemin.split("/")[-1].replace(".plt", "")

        for j, seg in enumerate(segments):
            if len(seg) < 2:
                continue

            couleur = COULEURS[couleur_idx % len(COULEURS)]
            couleur_idx += 1
            st = stats_segment(seg)

            coords = [(p["lat"], p["lon"]) for p in seg]

            # Ligne de trajectoire avec animation de direction (AntPath)
            AntPath(
                coords,
                color=couleur,
                weight=3,
                opacity=0.8,
                delay=800,
                dash_array=[10, 20],
                tooltip=f"{nom_fichier} — seg {j} | {st['n']} pts | "
                        f"{st['duree_min']} min | {st['dist_km']} km"
            ).add_to(carte)

            # Marqueur de depart (cercle vert)
            debut = seg[0]
            folium.CircleMarker(
                location=[debut["lat"], debut["lon"]],
                radius=7,
                color="green",
                fill=True,
                fill_color="green",
                fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>DEPART</b><br>{nom_fichier} seg {j}<br>"
                    f"{debut['label']}<br>{st['n']} pts | "
                    f"{st['duree_min']} min | {st['dist_km']} km",
                    max_width=220
                ),
                tooltip="Départ"
            ).add_to(carte)

            # Marqueur de fin (cercle rouge)
            fin = seg[-1]
            folium.CircleMarker(
                location=[fin["lat"], fin["lon"]],
                radius=7,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>ARRIVEE</b><br>{nom_fichier} seg {j}<br>"
                    f"{fin['label']}",
                    max_width=220
                ),
                tooltip="Arrivée"
            ).add_to(carte)

    # Legende simple (titre en haut a droite)
    titre_html = """
    <div style="position:fixed; top:10px; right:10px; z-index:1000;
                background:white; padding:10px 14px; border-radius:8px;
                border:1px solid #ccc; font-family:sans-serif; font-size:13px;">
        <b>Trajectoires GPS</b><br>
        <span style="color:green">●</span> Départ &nbsp;
        <span style="color:red">●</span> Arrivée<br>
        <i>Cliquer sur les marqueurs<br>pour les détails</i>
    </div>
    """
    carte.get_root().html.add_child(folium.Element(titre_html))

    return carte


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fichiers", nargs="+",
                        help="Fichier(s) .plt à tracer")
    parser.add_argument("--sortie", default="carte_trajectoire.html",
                        help="Nom du fichier HTML de sortie (défaut: carte_trajectoire.html)")
    args = parser.parse_args()

    carte = construire_carte(args.fichiers)
    if carte is None:
        sys.exit(1)

    carte.save(args.sortie)
    print(f"Carte enregistrée : {args.sortie}")
    print("Ouvrez ce fichier dans votre navigateur.")


if __name__ == "__main__":
    main()