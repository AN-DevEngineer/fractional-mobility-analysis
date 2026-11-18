# Littérature utilisée pour les arrêts et les lois de puissance

Sources consultées pour justifier les choix de `pretraitement/arrets.py`, `lois_de_puissance/` et `experiences/`. La colonne « Consulté » dit ce qui a
réellement été lu. **Les titres, volumes et pages saisis de mémoire sont à vérifier avant de les citer dans le mémoire.**

| Référence | Sert à | Consulté |
|---|---|---|
| Li et al. (2008), *Mining user similarity based on location history*, ACM GIS | Algorithme de détection d'arrêts par fenêtre glissante (`arrets.py`) | Décrit et cité par d'autres articles ; référence à vérifier |
| Zheng et al., travaux sur GeoLife | Seuils 200 m et 20-30 min | Via des articles qui les citent |
| Documentation de `trackintel` (prétraitement) | Valeurs par défaut 100 m / 5 min ; trous de plus de 15 min exclus de la génération d'arrêts | Documentation en ligne |
| Alessandretti, Sapiezynski, Lehmann, Baronchelli, arXiv:1609.05514 | Temps d'attente : log-normale en général, loi de puissance seulement dans la queue (au-delà de 13 h) ; arrêt = au moins 10 min ; tableau comparatif de plus de 30 jeux de données | Lu en entier (arXiv) |
| Wang, Han, Wang, arXiv:1303.5844, « Correlations and Scaling Laws in Human Mobility » | Référence sur GeoLife : séjours, distances, H par DFA | Lu en entier (arXiv) |
| Zhao et al., *Scientific Reports* 5:9136 (2015), « Explaining the power-law distribution of human mobility through transportation modality decomposition » | Chaque mode de transport log-normal ; leur mélange donne une loi de puissance | Page PMC (résumé et texte partiel) |
| Rhee et al. (2011), « On the Lévy-walk nature of human mobility », IEEE/ACM ToN | Pauses à queue lourde compatibles avec une loi de puissance, mais aussi Weibull ou log-normale | Résultats de recherche |
| Clauset, Shalizi, Newman (2009), « Power-law distributions in empirical data », SIAM Review 51(4), 661-703 | Méthode de `clauset.py` : exposant, x_min (distance KS), bootstrap, rapports de vraisemblance | Connue ; formules reprises dans les docstrings |
| Alstott, Bullmore, Plenz (2014), *PLoS ONE* 9(1):e85777, paquet `powerlaw` | Référence de validation de `clauset.py` | Pages PyPI et dépôt |
| Kivelä, Porter (2015), *Phys. Rev. E* 92, 052813 (arXiv:1412.8388) | Biais de la fenêtre d'observation finie sur les durées : séjours censurés | Résumé |
| Taqqu, Willinger, Sherman (1997), « Proof of a fundamental result in self-similar traffic modeling », ACM CCR | Relation entre l'exposant de queue des durées ON/OFF et H (hypothèse H2) | Résumé ; la forme H = (3 − a) / 2 est celle que je connais, **à confronter à la source** |
| arXiv:2404.17093 (titre à vérifier) | Autocorrélation d'un processus de renouvellement : décroissance en t^(−γ), γ = 2 − α pour 1 < α < 2 et γ = α − 2 pour 2 < α < 3 (α : exposant de densité) | Extrait de résultat de recherche |
| « Modelling bursty time series », *New J. Phys.* (2013), arXiv:1211.1175 | Une décroissance lente de l'autocorrélation n'indique une dépendance longue que si la relation d'échelle est violée | Résumé |
| Maraun, Rust, Timmer (2004), *Nonlin. Processes Geophys.* 11, 495-503 | Une pente DFA ne prouve pas la mémoire longue : modèles nuls à mémoire courte (`validation/modeles_nuls.py`) | Résumé |

## Pourquoi l'exposant de la formule H = (3 − a) / 2 n'est pas celui de Clauset

`clauset.py` estime l'exposant de la **densité** alpha (p(x) ~ x^(−alpha)). La formule de Taqqu et al. porte sur l'exposant de la **survie**
a (P(T > t) ~ t^(−a)), avec a = alpha − 1. La condition 1 < a < 2 devient 2 < alpha < 3, et H = (3 − a) / 2 = (4 − alpha) / 2.
Cette conversion est la mienne ; elle est cohérente avec la relation d'autocorrélation d'un processus de renouvellement (γ = alpha − 2 pour 2 < alpha < 3,
donc H = 1 − γ / 2 = (4 − alpha) / 2), mais elle est **à vérifier** avant publication.
