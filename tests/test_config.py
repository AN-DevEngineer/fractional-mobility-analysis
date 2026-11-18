import importlib

import config


def test_constantes_v4_inchangees():
    """Les constantes de la V4 gardent leur nom et leur valeur : rien ne change tant qu'on ne le decide pas."""
    assert config.TIME_GAP_THRESHOLD == 600
    assert config.MIN_POINTS == 300
    assert config.MAX_SPEED == 200.0
    assert config.DISTANCE_MIN == 0.1
    assert (config.N_S, config.N_M, config.M_MIN_RS, config.K_WHITTLE, config.N_MIN_WHITTLE) == (20, 20, 10, 1000, 20)


def test_valeurs_par_defaut_v5_reproduisent_la_v4():
    assert config.DFA_ECHELLE_MIN == 4            # la V4 fixait s_min = 4
    assert config.DFA_DIVISEUR_ECHELLE_MAX == 4
    assert config.RS_DIVISEUR_M_MAX == 2
    assert config.SERIE_ANALYSEE in {"distance", "vitesse"}


def test_empreinte_stable_et_sensible(monkeypatch):
    e1 = config.empreinte_config()
    assert e1 == config.empreinte_config() and len(e1) == 10
    monkeypatch.setattr(config, "NUL_NB_SURROGATS", config.NUL_NB_SURROGATS + 1)
    assert config.empreinte_config() != e1


def test_aucune_constante_orpheline_de_l_ancienne_api():
    assert not hasattr(config, "COUPER_AUX_ARTEFACTS")
