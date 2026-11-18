# Décisions à valider avec l'encadrant

Chaque décision est **réversible** par `config.py` (sauf mention). Principe général : **les valeurs par défaut reproduisent la V4**.
Rien ne change tant qu'une expérience ne l'a pas justifié.

| # | Décision | Raison | Comment revenir en arrière |
|---|---|---|---|
| 1 | **Les valeurs par défaut = V4** : DFA `s_min = 4`, série d_i, pas de coupure, MAD conservé | « La V5 est une évolution » : un fichier ou une valeur de la V4 garde son rôle | rien à faire |
| 2 | **Le marquage MAD est conservé** comme couche d'observation (points rouges des courbes, logs). Il ne retire rien | C'est votre démarche : regarder avant de décider | — |
| 3 | La **coupure aux artefacts n'est pas appliquée** : elle est mesurée. `03_estimation_H.py` écrit `h_dfa_morceaux` à côté de `h_dfa` ; `experiences/sensibilite_pretraitement.py` la compare à un **placebo** (même nombre de points retirés, positions aléatoires) | Résultat préliminaire (316 segments, 2 utilisateurs) : couper aux artefacts fait monter la médiane de H_DFA de 0,865 à 0,984, mais couper au hasard la fait déjà monter à 0,934. Une partie de l'effet est du simple raccourcissement | décider *après* l'expérience sur tout le corpus |
| 4 | Horodatages interprétés en **UTC** (`convertir_timestamp` déplacée dans `lecture.py`) | La V4 utilisait l'heure locale de la machine. Les Δt étaient justes, les timestamps absolus dépendaient du fuseau. Sans effet sur H, ni au Sénégal (UTC, sans heure d'été) | — |
| 5 | R/S : on garde `h_rs` (brut, colonne V4) et on **ajoute** `h_rs_corrige` (Anis-Lloyd, sans facteur de Peters) | Le R/S brut donne ~0,55 au lieu de 0,50 sur du bruit sans mémoire. Le facteur de Peters sous-estime l'espérance de 5 % à n = 10 pour cette définition (vérifié par Monte-Carlo) | colonne supplémentaire, rien n'est remplacé |
| 6 | Le « Whittle » est un Whittle **local** (Robinson) : **à corriger dans le texte** du chapitre 2 | C'est ce que fait réellement le code V4 | implémenter un Whittle paramétrique si l'encadrant le veut |
| 7 | Test nul AR(p) : **p = 1 et p = 5 tous deux rapportés** (`--ordre-ar`) | AR(5) : peu de faux positifs, peu de puissance (fGn H = 0,9, N = 1000 : 12 %). AR(1) : puissant (97 %). La conclusion dépend du choix, donc on montre les deux | `NUL_ORDRE_AR_MAX` |
| 8 | Mode d'enregistrement temps/distance par **heuristique** | Le guide GeoLife décrit deux modes ; il faut stratifier. Seuils `ECHANT_*` | **à valider sur le corpus complet** avec `00_audit_echantillonnage.py` |
| 9 | Série d'analyse d_i ; v_i en robustesse (`--serie vitesse`) | Cohérent avec l'analyse des distances du sujet | `SERIE_ANALYSEE` |
| 10 | **Logs et courbes : segments retenus seulement** (voir la section dédiée plus bas) | Les segments écartés ne servent ni à l'estimation ni à l'œil ; les journaliser et les tracer coûtait du temps de calcul et de l'espace disque | remettre `segments_info` à la place de `segments_retenus` dans `01_pretraitement.py` |
| 11 | **Noms en français partout** | Demandé. Piège pytest : une fonction nommée `test_…` ou `tester_…` est collectée comme un test ; elle s'appelle `evaluer_depassement_dfa` | — |
| 12 | **Arrêts : algorithme de Li et al., seuils 100 m / 5 min par défaut**, testés aussi à 100 m / 10 min, 200 m / 20 min et 200 m / 30 min | Standard de la littérature GPS : 100 m / 5 min = défaut de `trackintel`, 200 m / 20-30 min = Zheng et al. (très utilisé sur GeoLife). Les seuils sont arbitraires : une conclusion n'est retenue que si elle **résiste au changement de jeu** | `ARRET_RAYON_M`, `ARRET_DUREE_MIN_S`, `ARRET_JEUX_SENSIBILITE` |
| 13 | **Séjours censurés : étiquetés et exclus de l'ajustement principal** (arrêt qui commence au 1er point, arrêt jamais terminé, pause entre deux segments) | Une fenêtre d'observation finie tronque les longs séjours et biaise la queue (Kivelä et Porter, 2015). **Modifie une proposition précédente** qui ajoutait les pauses entre enregistrements aux temps de séjour | `temps_de_sejour(seulement_non_censurees=False)` ; variante `sejour_avec_censurees` de l'expérience |
| 14 | **Les distances sont celles entre arrêts consécutifs du même segment**, pas les d_i entre points GPS | Comme Wang et al. et Alessandretti et al. La variante à cheval sur un trou d'enregistrement est calculée à part (`distance_a_cheval`) : des arrêts intermédiaires ont pu passer inaperçus, ce qui gonfle les distances | `meme_segment_seulement` |
| 15 | **Clauset codé à la main, validé contre le paquet `powerlaw`** (test optionnel) | Vous avez choisi d'écrire les méthodes à la main. Le paquet sert de référence, jamais de dépendance du code | `pip install powerlaw` pour activer le test de comparaison |
| 16 | **La loi de puissance doit pouvoir être rejetée** : comparée à l'exponentielle, la log-normale, la Weibull et la loi tronquée ; verdict `rejetee` / `alternative_meilleure` / `plausible` | La littérature est partagée (log-normale pour les temps d'attente, loi tronquée, mélange de modes de transport). « Plausible » ne prouve rien | `LP_SEUIL_P` |
| 17 | **Le test de H2 vérifie d'abord le domaine de validité** : H = (3 − a) / 2 pour 1 < a < 2 (a = alpha − 1), soit 2 < alpha < 3 ; puis compare H observé à la théorie ET à H simulé à la même longueur | Les exposants de densité publiés pour les temps de séjour sont plutôt de 1,2 à 2,0 : peut-être hors domaine. DFA sous-estime H à N = 500 | si hors domaine, on rapporte que la prédiction ne s'applique pas |

## Ce qui a changé de place (rien n'a changé de contenu, sauf mention)

| V4 | V5 |
|---|---|
| `main.py` | `01_pretraitement.py` (identique, sauf : logs et courbes **des seuls segments retenus**, et deux lignes « segments écartés » ajoutées au rapport) |
| `estimation_H_all_seg.py` | `03_estimation_H.py` (13 colonnes V4 identiques + colonnes V5 ajoutées à la fin) |
| `pretraitement/utils.py` | `pretraitement/geodesie.py` ; `convertir_timestamp` -> `lecture.py` (UTC) |
| `stats_segment_gps.py` | `hurst/serie_distances.py` + `hurst/statistiques_segment.py` |
| `test_unitaire.py` | `tests/test_unitaire.py` (+ une ligne d'import) |
| `analyse_tdrive.py` | `experiences/analyse_tdrive.py` |
| `carte_trajectoire.py`, `diagnostic_log.py`, `log_pretraitement_csv.py`, `estimation_H.py` | `outils/` (+ une ligne d'import) |
| `hurst/dfa.py`, `rs.py`, `whittle.py` | identiques, avec les versions rapides **ajoutées à la suite** |
| `pretraitement/courbe.py` | image par défaut **identique octet pour octet** ; options ajoutées (`panneau_indice`, `h_estimes`, `tracer_courbe_serie`) |

## Observations sur le code V4 (non modifiées)

* `pretraitement/journal.py` définit `_construire_tableau_phases` **deux fois** : la seconde définition remplace la première.
* `outils/estimation_H.py` s'exécute **dès son import** (pas de garde `if __name__ == "__main__"`) et traite l'utilisateur 182 en dur.
* Dans `resultats_H/<user>_H.csv`, la colonne `segment_id` est la **position** du segment (0, 1, 2…), pas l'identifiant du CSV `sortie/`.
  Les deux coïncident tant que les identifiants de `sortie/` sont consécutifs ; `tests/test_parite_v4.py` en dépend.
* `outils/carte_trajectoire.py` demande le paquet `folium` (ajouté à `requirements.txt`).

## Non porté / pas encore fait

* Profilage (ACP, K-means) : non commencé ; il utilisera les exposants des temps de séjour et des distances (`resultats_H/lois_de_puissance/lois_resume.csv`).
* Cartographie des zones de mémoire : non commencée, la moins prioritaire (pré-empté dans sa forme générale par Tan et al., 2025).
* Une description à corriger dans le mémoire : le Whittle (point 6), le tableau 6.1 (voir `RESULTATS_PRELIMINAIRES.md` et l'audit PDF).

## Logs et courbes : seulement les segments retenus

**Ce qui a changé** (`01_pretraitement.py`) : `log_segments/<user>/log_segment_X.txt` et `courbe_distance_X.png` ne sont plus écrits pour les segments écartés
par le filtre final (trop courts, ou distance parcourue inférieure à `DISTANCE_MIN`). Les segments écartés sont seulement **comptés** : le rapport de chaque
utilisateur affiche « Segments écartés : N (< MIN_POINTS points : a ; distance < DISTANCE_MIN km : b) », ce qui aide à choisir ces deux seuils.

**Conséquence sur la numérotation.** Avant, les logs et les courbes étaient numérotés sur *tous* les segments produits (écartés compris), alors que
`sortie/<user>.csv` ne numérote que les retenus (`segment_id`). Les deux numérotations étaient décalées : `courbe_distance_9.png` pouvait être le segment
`segment_id = 3`. Maintenant, `log_segment_X.txt`, `courbe_distance_X.png` et `segment_id = X` désignent **le même segment**. Quand vous lisez une courbe pour
deviner H, vous comparez au bon segment dans `resultats_H/`.

**Reprise.** `main()` reconnaît un utilisateur « déjà traité » par l'existence de son dossier dans `log_segments/`. Ce dossier est donc toujours créé, même
vide ; sans cela, un utilisateur sans segment retenu serait retraité à chaque lancement.

**Utilisateurs déjà traités avant la correction.** Leurs anciens logs gardent l'ancienne numérotation et leurs segments écartés. Recommandation : ne pas tout
refaire maintenant. Quand `MIN_POINTS` et `TIME_GAP_THRESHOLD` seront figés, les segments retenus changeront de toute façon et il faudra tout relancer : ce sera
le moment de supprimer `log_segments/`.

**Vérification** sur un utilisateur traité après la correction : ces deux nombres doivent être égaux.
```
python -c "import pandas as pd; print(pd.read_csv('sortie/000.csv').segment_id.nunique())"
ls log_segments/000/log_segment_*.txt | wc -l
```

## À quoi sert `00_audit_echantillonnage.py`

Ce script n'existait pas en V4. Il ne calcule aucune valeur de H. Il lit les fichiers GeoLife tels qu'ils arrivent (lecture, nettoyage, tri, doublons), **avant** toute
segmentation, et décrit comment les points ont été enregistrés. Son but : **justifier les seuils par des chiffres** plutôt que par habitude. Il est facultatif, mais
le mémoire gagne à avoir ces chiffres.

| Décision | Ce que le CSV d'audit permet de lire | Limite |
|---|---|---|
| `TIME_GAP_THRESHOLD` (600 s) | Les colonnes `part_dt_sup_15s`, `part_dt_sup_60s`, `part_dt_sup_600s` : la part d'intervalles plus longs que 15 s, 60 s, 600 s. Si presque tous les pas durent quelques secondes, un trou de plusieurs minutes laissé *dans* un segment crée un saut énorme dans d_i, auquel DFA est très sensible. Ces parts disent combien de points on perd en coupant plus tôt | Décrit les fichiers, pas les segments |
| `MIN_POINTS` (300) | La distribution de `n_points` : quelle part des points disparaît à 100, 300 ou 500. Compromis entre quantité de données et précision de H (plus un segment est court, moins H est précis) | Fichiers entiers, alors que les segments sont plus courts après coupure : pour trancher, utiliser plutôt les lignes « Segments écartés » du rapport de `01_pretraitement.py` |
| Faut-il séparer les modes d'enregistrement ? | La colonne `mode` : « temps » (un point toutes les 1 à 5 s), « distance » (un point tous les 5 à 10 m) ou « mixte ». En mode distance, d_i est presque constant quelle que soit la vitesse : H décrit alors l'appareil. Si ce mode est rare, on peut l'ignorer ; s'il est important, il faut analyser les modes séparément | **Heuristique non validée** (seuils `ECHANT_*` de `config.py`) : vérifier à l'œil quelques fichiers de chaque mode |

Colonnes utiles en plus : `pas_median` (le rythme typique), `cv_dt` (irrégularité des pas), `corr_d_dt` (proche de 1 : d_i ne mesure que l'écart de temps, pas le déplacement).

## Correction à l'audit : le point C3 (numérotation des segments)

L'audit PDF signalait que les « segment 9 » et « segment 2 » du chapitre 7 du mémoire correspondaient aux segments 3 et 1 de la V4 (mêmes valeurs). La cause la plus
probable est la numérotation décalée des logs décrite ci-dessus : le mémoire citerait l'index du log, la V4 l'identifiant de `sortie/`. Les valeurs, elles, étaient bonnes.
Dans le mémoire, citer le `segment_id` de `sortie/`.

## Arrêts et lois de puissance (V5.1)

**Ce qui a été ajouté.**
`pretraitement/arrets.py` (détection d'arrêts, séjours, distances) ; `lois_de_puissance/` (Clauset, alternatives, analyse) ;
`experiences/lois_de_puissance.py` (l'ensemble sur GeoLife, avec sensibilité aux seuils d'arrêt) ; la partie réelle de
`experiences/mecanisme_on_off.py` (option `--reel`). Constantes dans `config.py`, section « AJOUTS V5.1 ». Références : `documentation/LITTERATURE.md`.

**Dans quel ordre les lancer** (après `01_pretraitement.py` et `03_estimation_H.py`) :
```
python experiences/lois_de_puissance.py --utilisateurs 000 001 --n-bootstrap 50 --n-bootstrap-utilisateur 0 --un-seul-jeu --sans-alternatives   # essai rapide
python experiences/lois_de_puissance.py                                                                                                    # corpus complet
python experiences/mecanisme_on_off.py --reel
```

**Comment lire `resultats_H/lois_de_puissance/lois_resume.csv`** : une ligne par (jeu de seuils, population, variable).
`verdict` : `rejetee` (p <= 0,1), `alternative_meilleure` (non rejetée mais une autre loi est significativement meilleure), `plausible`
(non rejetée, aucune alternative meilleure : les données ne l'écartent pas), `insuffisant`, `indetermine` (pas de p-valeur).
Regarder d'abord si le verdict et `alpha` changent d'un jeu de seuils à l'autre : si oui, la conclusion dépend d'un choix arbitraire.
`dans_domaine_on_off` indique si 2 < alpha < 3.

**Limites à connaître.**
* Entre deux segments *retenus* se trouvent parfois des segments *écartés* : une pause « entre_segments » peut alors cacher un déplacement.
* La recherche de x_min se limite à `LP_NB_CANDIDATS_XMIN` valeurs (le paquet `powerlaw` les essaie toutes) : écart négligeable en pratique.
* Le test du chi2 contre la loi tronquée est approché (la coupure nulle est au bord de l'espace des paramètres) : légèrement conservateur.
* Le modèle ON/OFF simulé suppose les durées de déplacement à queue plus légère que les arrêts et simule l'indicateur 0/1, pas les d_i.

**Statut de vérification à la livraison.** Le code a été **compilé** (syntaxe) et relu, mais les tests de cette étape
(`test_arrets.py`, `test_clauset.py`, `test_alternatives.py`, `test_analyse_distribution.py`, `test_experiences_lois.py`) **n'ont pas été exécutés** :
ils sont à lancer avec `python -m pytest` (les plus longs : `python -m pytest -m lent`). Un échec peut venir du code *ou* d'une tolérance
trop serrée dans un test statistique ; dans les deux cas, envoyer le message d'erreur.
