# ./lois_de_puissance/__init__.py
#
# Lois de puissance : methode de Clauset ecrite a la main, lois alternatives, analyse complete.
#
#   clauset.py       exposant (maximum de vraisemblance), seuil xmin (distance de Kolmogorov-Smirnov),
#                    p-valeur d'adequation (bootstrap semi-parametrique)
#   alternatives.py  exponentielle, log-normale, Weibull, loi de puissance tronquee ; rapports de
#                    vraisemblance (Vuong, chi2) et poids d'Akaike
#   analyse.py       analyser_distribution() : tout cela en une ligne de resultats, avec un verdict

from lois_de_puissance.analyse import analyser_distribution
from lois_de_puissance.clauset import (AjustementPuissance, ajuster_loi_de_puissance,
                                       exposant_vraisemblance_max, p_valeur_adequation,
                                       simuler_loi_de_puissance)

__all__ = ["analyser_distribution", "AjustementPuissance", "ajuster_loi_de_puissance",
           "exposant_vraisemblance_max", "p_valeur_adequation", "simuler_loi_de_puissance"]
