# ./lois_de_puissance/clauset.py
#
# AJUSTEMENT D'UNE LOI DE PUISSANCE : METHODE DE CLAUSET, SHALIZI ET NEWMAN (2009)
# "Power-law distributions in empirical data", SIAM Review 51(4), 661-703.
#
# Ecrit a la main (comme DFA, R/S et Whittle) pour que chaque etape soit lisible et explicable dans
# le memoire. Il est VALIDE contre le paquet `powerlaw` (Alstott, Bullmore & Plenz, 2014), qui
# implemente la meme methode : voir tests/test_clauset.py.
#
# ----------------------------------------------------------------------------------------
# Le modele
# ----------------------------------------------------------------------------------------
# On suppose que la QUEUE des donnees (x >= xmin) suit une loi de puissance continue :
#
#       p(x) = (alpha - 1) / xmin * (x / xmin) ^ (-alpha)          pour x >= xmin
#       P(X > x) = (x / xmin) ^ (1 - alpha)                         (fonction de survie)
#
# ATTENTION a deux exposants qu'on confond souvent :
#   alpha          exposant de la DENSITE   (celui de Clauset, et de la plupart des articles)
#   alpha - 1      exposant de la SURVIE    (celui de Taqqu et al. pour le lien avec H)
# La relation H = (3 - a) / 2 des processus ON/OFF porte sur a = alpha - 1, donc elle ne
# s'applique que pour 1 < a < 2, c'est-a-dire 2 < alpha < 3 (voir analyse.py).
#
# ----------------------------------------------------------------------------------------
# Les trois etapes de la methode
# ----------------------------------------------------------------------------------------
# 1. Pour un xmin donne, l'exposant du maximum de vraisemblance est
#       alpha = 1 + n / somme( ln(x_i / xmin) ),     ecart-type (alpha - 1) / racine(n)
#    avec n le nombre de points de la queue (x_i >= xmin).
# 2. xmin n'est pas connu : on essaie plusieurs valeurs et on garde celle qui minimise la distance
#    de Kolmogorov-Smirnov entre la queue observee et la loi ajustee.
# 3. Un bon ajustement n'est pas une preuve : on teste l'ADEQUATION. On simule des echantillons
#    qui SUIVENT la loi ajustee (queue) et reprennent les donnees observees (corps), on refait tout
#    l'ajustement sur chacun, et la p-valeur est la part des simulations dont la distance KS est
#    au moins aussi grande que celle des donnees. Regle de Clauset : p <= 0.1 => loi de puissance
#    REJETEE ; p > 0.1 => PLAUSIBLE (pas prouvee : d'autres lois peuvent aussi convenir,
#    voir alternatives.py).
#
# Approximation continue : les durees (secondes) et les distances (metres) sont traitees comme des
# variables continues, ce qui est correct des que xmin est de l'ordre de la dizaine d'unites.

from dataclasses import dataclass

import numpy as np

from config import LP_NB_CANDIDATS_XMIN, LP_NB_MIN_QUEUE, LP_NB_BOOTSTRAP


# ---------------------------------------------------------------------------------------
# Brique 1 : exposant a xmin fixe
# ---------------------------------------------------------------------------------------
def exposant_vraisemblance_max(queue, xmin):
    """
    Exposant alpha de la loi de puissance continue, par maximum de vraisemblance.

        alpha = 1 + n / somme( ln(x_i / xmin) )

    Entree :
        queue : valeurs x_i >= xmin (tableau)
        xmin  : seuil (> 0)

    Sortie :
        (alpha, ecart_type) ; (nan, nan) si moins de 2 points ou si toutes les valeurs valent xmin.
    """
    queue = np.asarray(queue, dtype=float)
    n = len(queue)
    if n < 2:
        return float("nan"), float("nan")
    somme = float(np.sum(np.log(queue / xmin)))
    if somme <= 0:
        return float("nan"), float("nan")
    alpha = 1.0 + n / somme
    return float(alpha), float((alpha - 1.0) / np.sqrt(n))


# ---------------------------------------------------------------------------------------
# Brique 2 : distance de Kolmogorov-Smirnov
# ---------------------------------------------------------------------------------------
def distance_ks(queue_triee, alpha, xmin):
    """
    Distance de Kolmogorov-Smirnov entre la queue observee et la loi de puissance ajustee :
    plus grand ecart absolu entre les deux fonctions de repartition.

    Entree :
        queue_triee : valeurs x >= xmin, triees par ordre CROISSANT
        alpha, xmin : parametres de la loi ajustee

    Sortie : float entre 0 (accord parfait) et 1.

    On compare la repartition theorique a l'empirique AVANT et APRES chaque marche de l'escalier
    (i-1)/n et i/n : le maximum de l'ecart se trouve toujours a l'une de ces deux hauteurs.
    """
    n = len(queue_triee)
    theorique = 1.0 - (queue_triee / xmin) ** (1.0 - alpha)
    apres = np.arange(1, n + 1) / n
    avant = np.arange(0, n) / n
    return float(max(np.max(np.abs(apres - theorique)), np.max(np.abs(avant - theorique))))


# ---------------------------------------------------------------------------------------
# Brique 3 : choix de xmin
# ---------------------------------------------------------------------------------------
def _candidats_xmin(x_trie, nb_candidats, nb_min_queue):
    """
    Seuils xmin essayes : valeurs distinctes observees dont la queue a au moins nb_min_queue points,
    reduites a nb_candidats valeurs equireparties si elles sont trop nombreuses.

    Le paquet `powerlaw` essaie TOUTES les valeurs distinctes ; limiter a quelques centaines change
    tres peu le resultat et evite un cout quadratique sur de gros echantillons (le bootstrap refait
    la recherche des centaines de fois).
    """
    n = len(x_trie)
    distincts = np.unique(x_trie)
    # taille de la queue si xmin = v : nombre de valeurs >= v
    taille_queue = n - np.searchsorted(x_trie, distincts, side="left")
    valides = distincts[taille_queue >= nb_min_queue]
    if len(valides) > nb_candidats:
        positions = np.unique(np.linspace(0, len(valides) - 1, nb_candidats).astype(int))
        valides = valides[positions]
    return valides


@dataclass(frozen=True)
class AjustementPuissance:
    """Resultat d'un ajustement de loi de puissance (xmin choisi par minimisation de la distance KS)."""
    n: int                 # nombre total de points positifs
    xmin: float            # seuil retenu
    alpha: float           # exposant de la densite
    sigma_alpha: float     # ecart-type de alpha
    ks: float              # distance KS de l'ajustement retenu
    n_queue: int           # nombre de points x >= xmin


def ajuster_loi_de_puissance(x, nb_candidats=LP_NB_CANDIDATS_XMIN, nb_min_queue=LP_NB_MIN_QUEUE):
    """
    Ajuste une loi de puissance continue a la queue de x, avec xmin choisi par la methode de Clauset :
    pour chaque seuil candidat, exposant par maximum de vraisemblance puis distance KS ; on garde le
    seuil de distance KS minimale.

    Entree :
        x            : donnees (les valeurs <= 0 sont ignorees)
        nb_candidats : nombre maximal de seuils essayes
        nb_min_queue : taille minimale de la queue

    Sortie :
        AjustementPuissance, ou None si l'ajustement est impossible (trop peu de donnees).

    Fonctionnement : les donnees sont triees une fois ; la somme des ln(x_i) des queues se calcule
    par sommes cumulees depuis la fin, donc l'exposant de chaque seuil candidat coute O(1) ; seule la
    distance KS demande de parcourir la queue.
    """
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x > 0]
    n = len(x)
    if n < nb_min_queue:
        return None
    log_x = np.log(x)
    somme_logs_queue = np.cumsum(log_x[::-1])[::-1]       # somme_logs_queue[k] = somme des ln x_i pour i >= k
    meilleur = None
    for xmin in _candidats_xmin(x, nb_candidats, nb_min_queue):
        k = int(np.searchsorted(x, xmin, side="left"))     # premier indice de la queue
        n_queue = n - k
        somme = somme_logs_queue[k] - n_queue * np.log(xmin)
        if somme <= 0:
            continue
        alpha = 1.0 + n_queue / somme
        ks = distance_ks(x[k:], alpha, xmin)
        if meilleur is None or ks < meilleur.ks:
            meilleur = AjustementPuissance(n=n, xmin=float(xmin), alpha=float(alpha),
                                           sigma_alpha=float((alpha - 1.0) / np.sqrt(n_queue)),
                                           ks=float(ks), n_queue=int(n_queue))
    return meilleur


# ---------------------------------------------------------------------------------------
# Brique 4 : simulation et adequation
# ---------------------------------------------------------------------------------------
def simuler_loi_de_puissance(n, alpha, xmin, rng):
    """
    n tirages d'une loi de puissance continue (alpha, xmin), par inversion de la fonction de repartition :
    x = xmin * (1 - U) ^ (-1 / (alpha - 1)) avec U uniforme sur [0, 1[.
    """
    return xmin * (1.0 - rng.random(n)) ** (-1.0 / (alpha - 1.0))


def p_valeur_adequation(x, ajustement, rng, nb_bootstrap=LP_NB_BOOTSTRAP,
                        nb_candidats=LP_NB_CANDIDATS_XMIN, nb_min_queue=LP_NB_MIN_QUEUE):
    """
    p-valeur d'adequation de la loi de puissance, par bootstrap semi-parametrique (Clauset et al.).

    Pour chaque simulation :
      1. on tire le nombre de points de queue selon la proportion observee (n_queue / n) ;
      2. la queue est SIMULEE selon la loi de puissance ajustee (alpha, xmin) ;
      3. le corps (valeurs < xmin) est REECHANTILLONNE dans les donnees observees : on ne suppose
         rien sur lui, il ne sert qu'a faire un echantillon de meme forme que les donnees ;
      4. on refait tout l'ajustement (y compris le choix de xmin : c'est ce qui rend la p-valeur
         honnete) et on note la distance KS.

    p = part des simulations dont la distance KS est >= celle des donnees.
    Lecture : p grand = les donnees ressemblent a ce que donnerait une vraie loi de puissance ;
    p <= 0.1 = on rejette (Clauset). Une p-valeur precise a 0.01 pres demande environ 1000 simulations.

    Entree :
        x           : donnees observees
        ajustement  : AjustementPuissance obtenu sur x par ajuster_loi_de_puissance()
        rng         : numpy.random.Generator (la reproductibilite en depend)

    Sortie : p-valeur (float) ; nan si aucune simulation n'a pu etre ajustee.
    """
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    n = len(x)
    corps = x[x < ajustement.xmin]
    proportion_queue = ajustement.n_queue / n
    superieurs = 0
    valides = 0
    for _ in range(nb_bootstrap):
        n_queue_sim = int(rng.binomial(n, proportion_queue))
        queue_sim = simuler_loi_de_puissance(n_queue_sim, ajustement.alpha, ajustement.xmin, rng)
        n_corps_sim = n - n_queue_sim
        if n_corps_sim > 0 and len(corps) > 0:
            corps_sim = rng.choice(corps, size=n_corps_sim, replace=True)
        else:
            corps_sim = np.empty(0)
        simule = ajuster_loi_de_puissance(np.concatenate([corps_sim, queue_sim]), nb_candidats, nb_min_queue)
        if simule is None:
            continue
        valides += 1
        if simule.ks >= ajustement.ks:
            superieurs += 1
    return float(superieurs / valides) if valides > 0 else float("nan")
