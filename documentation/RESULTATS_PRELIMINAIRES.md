# Résultats préliminaires (exploration du 18 octobre 2026)

**Statut : exploration, pas résultats du mémoire.** Peu de données (utilisateurs 000 et 001 uniquement, 316 segments), seuils non figés,
60 à 150 répétitions par cellule de simulation. À régénérer avec les scripts après validation de `DECISIONS.md`.
Réglages : `config.py` par défaut (= V4), graine 20261018.

## 1. Parité avec la V4

* `03_estimation_H.py` sur vos CSV `sortie/` des utilisateurs 000 et 001 : les **13 colonnes V4** sont identiques à vos `resultats_H/` à 8·10⁻¹¹ près
  (316 segments), avec les mêmes segments marqués SUSPECT.
* L'image par défaut de `courbe.py` est identique octet pour octet à celle de la V4.

## 2. Exactitude des estimateurs sur fGn (RMSE ; 150 répétitions ; `02_validation_methodes.py`)

| H₀ | N | DFA | R/S brut | R/S corrigé | Whittle local |
|---|---|---|---|---|---|
| 0,5 | 100 | 0,108 | 0,141 | 0,103 | 0,151 |
| 0,5 | 1000 | 0,035 | 0,065 | 0,036 | 0,057 |
| 0,7 | 300 | 0,063 | 0,060 | 0,087 | 0,100 |
| 0,9 | 100 | 0,120 | 0,144 | 0,211 | 0,116 |
| 0,9 | 1000 | 0,045 | 0,075 | 0,125 | 0,059 |

Aucun estimateur ne domine partout. DFA a le plus petit RMSE dans 7 des 12 cellules (H₀ de 0,3 à 0,9, N de 100 à 1000) ; le R/S corrigé retire le biais à H = 0,5 mais sous-estime
les séries persistantes. **Ce tableau remplace le tableau 6.1 du mémoire** (non reproductible : voir l'audit PDF).

## 3. Test nul : taux de rejet à α = 5 % (60 répétitions par cellule, donc ± 0,05)

| Série simulée | N | nul AR(1) | nul AR(≤5) |
|---|---|---|---|
| AR(1) φ = 0,5 (mémoire courte) | 1000 | 0,05 | 0,00 |
| i.i.d. | 1000 | 0,02 | 0,05 |
| fGn H = 0,7 (mémoire longue) | 1000 | 0,92 | 0,20 |
| fGn H = 0,9 (mémoire longue) | 1000 | 0,97 | 0,12 |
| ON/OFF Pareto α = 1,5 (H théorique 0,75) | 1000 | 0,38 | 0,00 |
| Changements de régime | 1000 | 1,00 | 0,22 |

Lecture : avec AR(≤5) le test est conservateur mais presque aveugle à N ≤ 1000. Avec AR(1) il est puissant et bien calibré sur ces séries
(5 à 7 % de faux positifs sur AR(1)). **Il ne sépare pas mémoire longue et changements de régime.**
*Correction d'une première estimation* faite avec `s_min = 10` : elle donnait 13 % de faux positifs pour AR(1) ; avec le défaut V4 (`s_min = 4`)
on obtient 5 à 7 %. À trancher avec ≥ 200 répétitions avant de l'écrire dans le mémoire.

## 4. Vos segments réels (utilisateurs 000 et 001, 316 segments)

| | Résultat |
|---|---|
| H médian : DFA / R/S brut / R/S corrigé / Whittle | 0,865 / 0,823 / 0,758 / 0,858 |
| DFA > 1 | 29,7 % |
| Whittle à la borne de la grille | 33,5 % |
| Segments SUSPECT (définition V4) | 61,1 % |
| Dépassement du modèle nul AR(≤5) / AR(1) | 27,5 % / 42,7 % |
| Modes d'enregistrement détectés | temps 215, distance 78, mixte 23 |
| H_DFA médian : temps / distance / mixte | 0,879 / 0,803 / 0,870 |

Lecture prudente : entre 28 % et 43 % des segments dépassent un modèle à mémoire courte selon le nul retenu ; les autres ne sont pas
distinguables d'une mémoire courte avec cette puissance.

## 5. Sensibilité de H au prétraitement (`experiences/sensibilite_pretraitement.py`, mêmes 316 segments)

| Variante | H_DFA médian | écart absolu médian vs base | part des segments où l'écart > 0,1 | DFA > 1 |
|---|---|---|---|---|
| base (V4) | 0,865 | — | — | 29,7 % |
| coupure aux artefacts | 0,984 | 0,045 | 41,4 % | 46,5 % |
| **placebo** (même nombre de points retirés, au hasard) | 0,934 | 0,028 | 32,4 % | 37,5 % |
| coupure aux trous > 60 s | 0,938 | 0,031 | 31,0 % | 40,2 % |
| coupure aux intervalles > 3 × le pas médian | 0,980 | 0,062 | 38,9 % | 45,6 % |
| série vitesse v_i au lieu de d_i | 0,943 | 0,056 | 37,7 % | 43,7 % |
| DFA `s_min = 10` | 0,825 | 0,053 | 22,2 % | 26,9 % |

Lecture : des **coupures aléatoires** modifient déjà H de plus de 0,1 pour un segment sur trois : H est instable face à des perturbations bénignes.
Les coupures aux artefacts vont plus loin que le placebo (+0,05 de médiane), donc une partie de l'effet leur est propre.

## 6. Lien ON/OFF (hypothèse H2), partie synthétique (`experiences/mecanisme_on_off.py`)

H estimé sur des ON/OFF à durées de Pareto, biais par rapport à (3 − α)/2 :

| α | H théorique | DFA, N = 3000 | DFA, N = 500 | Whittle, N = 3000 | Whittle, N = 500 |
|---|---|---|---|---|---|
| 1,2 | 0,90 | −0,011 | −0,101 | +0,069 | +0,011 |
| 1,4 | 0,80 | −0,014 | −0,088 | +0,098 | +0,039 |
| 1,6 | 0,70 | +0,004 | −0,063 | +0,121 | +0,074 |
| 1,8 | 0,60 | +0,021 | −0,043 | +0,150 | +0,125 |

DFA retrouve la théorie à ±0,02 quand N = 3000, mais **sous-estime de 0,04 à 0,10 à N = 500** (taille typique d'un segment) : le test H ↔ α sur
vos segments devra comparer H observé à la distribution simulée **à la même longueur**, pas à la valeur théorique brute.
