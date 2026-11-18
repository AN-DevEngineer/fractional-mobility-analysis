# ./simulation/__init__.py

from simulation.bruit_fractionnaire import bruit_gaussien_fractionnaire
from simulation.memoire_courte import ar1, changements_de_regime
from simulation.on_off_pareto import on_off


def catalogue_modeles():
    """Modeles de serie utilises pour mesurer faux positifs et puissance : nom -> f(n, rng)."""
    return {
        "iid":        lambda n, rng: rng.standard_normal(n),
        "ar1_0.5":    lambda n, rng: ar1(n, 0.5, rng),
        "ar1_0.8":    lambda n, rng: ar1(n, 0.8, rng),
        "regimes":    lambda n, rng: changements_de_regime(n, rng),
        "onoff_1.5":  lambda n, rng: on_off(n, 1.5, 1.5, rng),
        "fgn_0.6":    lambda n, rng: bruit_gaussien_fractionnaire(n, 0.6, rng),
        "fgn_0.7":    lambda n, rng: bruit_gaussien_fractionnaire(n, 0.7, rng),
        "fgn_0.9":    lambda n, rng: bruit_gaussien_fractionnaire(n, 0.9, rng),
    }
