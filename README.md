# Mémoire 1 — Dynamique fractionnaire de la mobilité urbaine (V5)

Mesurer **de façon fiable** la dépendance longue (exposant de Hurst *H*) dans les traces GPS de mobilité (GeoLife).
La V5 est une **évolution de la V4** : mêmes fichiers, mêmes noms, mêmes résultats (parité numérique vérifiée), avec en plus
des outils pour **tester** la mémoire longue au lieu de la présupposer.

## Installation

```bash
pip install -r requirements.txt
python -m pytest              # ~115 tests, quelques secondes
```

## Déroulé de travail

| Étape | Commande | Données requises |
|---|---|---|
| 0. Auditer l'échantillonnage, **choisir les seuils** | `python 00_audit_echantillonnage.py` | GeoLife dans `dataset/` |
| 1. Prétraitement (= V4 `main.py` ; logs et courbes des seuls segments retenus) | `python 01_pretraitement.py` | GeoLife |
| 2. Valider les méthodes sur séries de H connu | `python 02_validation_methodes.py --n-rep 500` | aucune |
| 3. Estimer H sur tous les segments (= V4 + colonnes) | `python 03_estimation_H.py` | `sortie/` |
| 4. Arrêts, temps de séjour, lois de puissance (Clauset) | `python experiences/lois_de_puissance.py` | `sortie/` |
| 5. Lien entre H et les temps de séjour (ON/OFF) | `python experiences/mecanisme_on_off.py --reel` | étapes 3 et 4 |
| Autres expériences | `experiences/sensibilite_pretraitement.py`, `validation_visuelle.py` | `sortie/` |

Options de `03_estimation_H.py` : `--sans-nul` (rapide), `--serie distance|vitesse`, `--ordre-ar 1|5`, `--sortie`, `--resultats`, `--logs`.
Chaque résultat porte `empreinte_config` : l'empreinte de **tous** les réglages de `config.py` au moment du calcul.

## Arborescence

```
memoire_M1/
├── config.py                        tous les seuils et réglages (constantes V4 inchangées + sections « AJOUTS V5 » et « AJOUTS V5.1 »)
├── 00_audit_echantillonnage.py      pas de temps, mode d'enregistrement : choisir les seuils
├── 01_pretraitement.py              (ex main.py) GeoLife brut -> segments marqués ; logs et courbes des segments retenus
├── 02_validation_methodes.py        validation Monte-Carlo ; tableaux LaTeX générés
├── 03_estimation_H.py               (ex estimation_H_all_seg.py) H par segment, 13 colonnes V4 + colonnes V5
│
├── pretraitement/
│   ├── lecture.py                   lecture des .plt ; convertir_timestamp (UTC)
│   ├── nettoyage.py   tri.py        points invalides, doublons, ordre temporel
│   ├── segmentation.py              coupure aux trous de temps ; couper_aux_points_marques (option V5)
│   ├── filtres.py                   marquage MAD des points suspects (observation, ne retire rien)
│   ├── echantillonnage.py           mode d'enregistrement : temps, distance ou mixte
│   ├── arrets.py                    détection d'arrêts (Li et al.), temps de séjour, distances entre arrêts
│   ├── stats_segment.py             statistiques de Phase 2 (stationnarité, hétérogénéité)
│   ├── geodesie.py                  (ex utils.py) haversine, vitesse, écart de temps
│   ├── sauvegarde.py
│   ├── journal.py                   logs texte par segment
│   └── courbe.py                    courbes par segment (+ options : 2 panneaux, H estimés)
│
├── hurst/
│   ├── dfa.py  rs.py  whittle.py    V4 (référence lisible, écrite à la main) + versions rapides numpy
│   ├── echelles.py                  grilles d'échelles, régressions log-log
│   ├── serie_distances.py           (ex stats_segment_gps) distances d_i, profil cumulé
│   ├── statistiques_segment.py      (ex stats_segment_gps) statistiques d'un segment
│   └── estimation_segment.py        assemble les 3 méthodes + test nul + diagnostics, par segment
│
├── simulation/                      séries de vérité connue
│   ├── bruit_fractionnaire.py       fGn et fBm
│   ├── memoire_courte.py            AR(1), changements de régime
│   ├── on_off_pareto.py             alternance mouvement/arrêt à durées de Pareto
│   └── marginale_lognormale.py
│
├── validation/
│   ├── monte_carlo.py               biais, RMSE, taux de rejet, tableaux LaTeX
│   └── modeles_nuls.py              surrogats à mémoire courte, test de dépassement
│
├── lois_de_puissance/               méthode de Clauset écrite à la main
│   ├── clauset.py                   exposant (max. de vraisemblance), x_min (distance KS), p-valeur (bootstrap)
│   ├── alternatives.py              exponentielle, log-normale, Weibull, loi tronquée ; Vuong, chi2, poids d'Akaike
│   └── analyse.py                   analyser_distribution() : une ligne de résultats et un verdict
│
├── experiences/                     une expérience par question de recherche
│   ├── sensibilite_pretraitement.py H3 : H selon les seuils, la série, le mode d'enregistrement (avec placebo)
│   ├── lois_de_puissance.py         arrêts, séjours et distances de GeoLife ; sensibilité aux seuils d'arrêt
│   ├── mecanisme_on_off.py          H2 : lien H <-> temps de séjour (partie 1 synthétique ; partie 2 --reel sur GeoLife)
│   ├── validation_visuelle.py       votre lecture des courbes, à l'aveugle, contre les estimateurs
│   └── analyse_tdrive.py            (V4)
│
├── tests/                           tests automatiques (pytest) + votre test manuel
│   ├── conftest.py
│   ├── test_config.py  test_lecture.py  test_nettoyage.py  test_segmentation.py  test_filtres.py
│   ├── test_echantillonnage.py  test_journal.py  test_courbe.py
│   ├── test_dfa.py  test_rs.py  test_whittle.py          (la référence = votre code V4 lui-même)
│   ├── test_simulation.py  test_modeles_nuls.py  test_monte_carlo.py
│   ├── test_estimation_segment.py  test_experiences.py  test_experiences_lois.py
│   ├── test_arrets.py  test_clauset.py  test_alternatives.py  test_analyse_distribution.py
│   ├── test_parite_v4.py            retrouve VOS résultats V4 réels (sortie/ + resultats_H/)
│   └── test_unitaire.py             (V4) test manuel d'un fichier : python tests/test_unitaire.py --file <plt>
│
├── outils/                          V4, inchangés (sauf la ligne d'import de la racine)
│   └── carte_trajectoire.py  diagnostic_log.py  log_pretraitement_csv.py  estimation_H.py
│
└── dataset/  sortie/  log_segments/  log_estimation_H/  resultats_H/  teste_pret/  images/
    memoire/  papiers/  documentation/
```

## Deux sortes de « tests »

* **`python -m pytest`** : tests automatiques du code. Ils vérifient que chaque module fait ce qu'il doit, et que les versions
  rapides redonnent les valeurs de votre code V4. À lancer après chaque modification.
* **`python tests/test_unitaire.py --file <plt>`** : votre test manuel V4 (un fichier, journal et courbes dans `teste_pret/`).

## Votre lecture des courbes

`pretraitement/courbe.py` trace d_i en fonction du **temps réel** (comme en V4, image inchangée par défaut). Les estimateurs, eux, reçoivent
d_i en fonction du **numéro de point**. Quand Δt est irrégulier, les deux images diffèrent. `COURBE_DEUX_PANNEAUX = True` dans `config.py`
(ou `panneau_indice=True`) ajoute le second panneau. `experiences/validation_visuelle.py` mesure, à l'aveugle, ce que votre œil lit vraiment.

## Limites à connaître

* Un exposant DFA > 1 sort du domaine du bruit gaussien fractionnaire : ce n'est pas un « H », il signale une série non stationnaire.
* Le test nul dit « non expliqué par une mémoire courte AR(p) », pas « mémoire longue » : des changements de régime le déclenchent aussi.
  Sa puissance dépend fortement de *p* et de *N* : présenter **p = 1 et p = 5**.
* Le « Whittle » du code est un Whittle **local** (basses fréquences), pas un Whittle sur [−π, π] : le chapitre 2 du mémoire est à corriger.
* L'heuristique du mode d'enregistrement n'est pas validée sur le corpus complet.
* `log_segment_X.txt`, `courbe_distance_X.png` et `segment_id = X` (dans `sortie/` et `resultats_H/`) désignent le même segment, pour les utilisateurs traités après la correction (voir `documentation/DECISIONS.md`).
* Les **lois de puissance** de `lois_de_puissance/` sont testées pour être **rejetées** : la littérature est partagée (log-normale, loi tronquée, mélange de modes
  de transport). Un verdict « plausible » ne prouve pas la loi de puissance, il veut dire que les données ne permettent pas de l'écarter.
* Le lien **H = (3 − a) / 2** (processus ON/OFF) porte sur l'exposant de **survie** a = alpha − 1 et n'est valable que pour 1 < a < 2, soit 2 < alpha < 3 pour l'exposant
  de **densité** de Clauset. Les exposants publiés pour les temps de séjour sont plutôt de 1,2 à 2,0 : vérifier le domaine avant de comparer à H
  (`experiences/mecanisme_on_off.py --reel` le fait).
* Les séjours dont le début ou la fin est inconnu (début de fichier, fin de fichier, pause entre deux enregistrements) sont **étiquetés censurés** et exclus de
  l'ajustement principal : une fenêtre d'observation finie tronque les longs séjours et biaise la queue (Kivelä et Porter, 2015).
