# ./lois_de_puissance/alternatives.py
#
# LOIS ALTERNATIVES ET COMPARAISON A LA LOI DE PUISSANCE
#
# Ajuster une loi de puissance qui "marche bien" ne prouve rien : beaucoup de lois ont une queue qui
# ressemble a une droite en echelles log-log. La litterature sur la mobilite humaine est d'ailleurs
# partagee (Alessandretti et al. : log-normale pour les temps d'attente ; Zhao et al. : chaque mode
# de transport est log-normal et leur melange donne une loi de puissance ; Rhee et al. : loi de
# puissance tronquee). Ce module met la loi de puissance en concurrence avec quatre alternatives,
# TOUTES ajustees sur la meme queue (x >= xmin, xmin fixe par clauset.py) :
#
#   exponentielle        p(x) = lambda * exp(-lambda * (x - xmin))                  1 parametre
#   lognormale           log-normale conditionnee a x >= xmin                       2 parametres
#   weibull              exponentielle etiree, conditionnee a x >= xmin             2 parametres
#   puissance_tronquee   p(x) ~ x^(-alpha) * exp(-lambda * x)  (loi de puissance a coupure
#                        exponentielle ; contient la loi de puissance pour lambda = 0)  2 parametres
#
# "Conditionnee a x >= xmin" : chaque loi est renormalisee pour ne decrire que la queue, ce qui rend
# les vraisemblances directement comparables a celle de la loi de puissance.
#
# ----------------------------------------------------------------------------------------
# Deux outils de comparaison (Clauset et al., 2009, section 5)
# ----------------------------------------------------------------------------------------
# 1. RAPPORT DE VRAISEMBLANCE  R = somme( ln p_puissance(x_i) - ln p_alternative(x_i) ).
#    R > 0 favorise la loi de puissance, R < 0 l'alternative. Le signe ne suffit pas : il faut
#    savoir s'il est SIGNIFICATIF. Pour deux modeles non emboites on utilise le test de Vuong
#    (p = erfc(|R| / (racine(2 n) * sigma)), sigma = ecart-type des differences point par point).
#    Pour la loi de puissance tronquee, qui CONTIENT la loi de puissance, on utilise le test du
#    chi2 a un degre de liberte sur 2|R| (approximation : le parametre lambda = 0 est au bord de
#    l'espace des parametres, ce qui rend le test un peu conservateur).
#    Un p > LP_SEUIL_P veut dire "indecidable" : les donnees ne tranchent pas.
# 2. POIDS D'AKAIKE (AICc) : probabilite relative que chaque modele soit le meilleur du lot,
#    en penalisant le nombre de parametres. Ils somment a 1.
#
# ----------------------------------------------------------------------------------------
# Adequation de la tronquee (AJOUT 1)
# ----------------------------------------------------------------------------------------
# La comparaison AICc dit "quel modele est le meilleur", pas "quel modele est ADEQUAT".
# Un modele peut gagner l'AICc et etre rejete par un test d'adequation (bootstrap KS).
# Le bloc en fin de fichier ajoute ce test, calque sur celui de Clauset pour la loi de puissance,
# pour la loi tronquee. Cout : ~200 simulations + ajustement a chaque fois.
# Resultat : p > 0.1 => tronquee non rejetee (modele plausible), p <= 0.1 => rejetee.
# Approximation assumee : x_min est fixe a sa valeur observee (pas re-choisi a chaque simulation),
# ce qui divise le cout par ~3 et reste valide conditionnellement a x_min.

from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize, special, stats

from config import LP_SEUIL_P

NOMS_ALTERNATIVES = ("exponentielle", "lognormale", "weibull", "puissance_tronquee")
NOMS_MODELES = ("loi_de_puissance",) + NOMS_ALTERNATIVES

_PENALITE = 1e300   # valeur de l'objectif quand les parametres proposes donnent une vraisemblance invalide


@dataclass(frozen=True)
class AjustementAlternatif:
    """Une loi alternative ajustee sur la queue."""
    nom: str
    parametres: dict          # parametres estimes (ex. {"mu": ..., "sigma": ...})
    logv_points: np.ndarray   # log-vraisemblance de CHAQUE point de la queue (pour le test de Vuong)
    nb_parametres: int        # utilise par le critere d'Akaike


# ---------------------------------------------------------------------------------------
# Loi de puissance (reference)
# ---------------------------------------------------------------------------------------
def logv_loi_de_puissance(queue, xmin, alpha):
    """ln p(x) = ln(alpha - 1) - ln(xmin) - alpha * ln(x / xmin), pour chaque point de la queue."""
    queue = np.asarray(queue, dtype=float)
    return np.log(alpha - 1.0) - np.log(xmin) - alpha * np.log(queue / xmin)


def _ajustement_valide(logv):
    """Vrai si toutes les log-vraisemblances sont des nombres finis (sinon l'ajustement est inutilisable)."""
    return bool(np.all(np.isfinite(logv)))


# ---------------------------------------------------------------------------------------
# Exponentielle
# ---------------------------------------------------------------------------------------
def ajuster_exponentielle(queue, xmin):
    """
    Exponentielle decalee : p(x) = lambda * exp(-lambda * (x - xmin)) pour x >= xmin.
    Estimateur du maximum de vraisemblance explicite : lambda = 1 / moyenne(x - xmin).
    Renvoie None si toutes les valeurs valent xmin.
    """
    decalage = np.asarray(queue, dtype=float) - xmin
    moyenne = float(decalage.mean())
    if not moyenne > 0:
        return None
    lam = 1.0 / moyenne
    logv = np.log(lam) - lam * decalage
    if not _ajustement_valide(logv):
        return None
    return AjustementAlternatif("exponentielle", {"lambda": lam}, logv, 1)


# ---------------------------------------------------------------------------------------
# Log-normale conditionnee
# ---------------------------------------------------------------------------------------
def _logv_lognormale(queue, xmin, mu, sigma):
    """
    ln p(x | x >= xmin) pour une log-normale (mu, sigma) :
        ln f(x) - ln P(X >= xmin)
    avec f la densite log-normale et P(X >= xmin) = 1 - Phi((ln xmin - mu) / sigma) (fonction de survie
    de la loi normale, calculee en logarithme pour rester precise loin dans la queue).
    """
    z = (np.log(queue) - mu) / sigma
    borne = (np.log(xmin) - mu) / sigma
    return (-np.log(queue) - np.log(sigma) - 0.5 * np.log(2.0 * np.pi) - 0.5 * z ** 2
            - stats.norm.logsf(borne))


def ajuster_lognormale(queue, xmin):
    """
    Log-normale conditionnee a x >= xmin, par maximum de vraisemblance numerique (Nelder-Mead sur
    (mu, ln sigma), departs pris sur les moments de ln x). Renvoie None si l'optimisation echoue.

    La troncature rend cette vraisemblance parfois tres plate (mu et sigma mal identifies quand la
    queue ne montre qu'une petite partie de la loi) : le test de Vuong en tient compte, car la
    variance des differences point par point est grande dans ce cas.
    """
    queue = np.asarray(queue, dtype=float)
    logs = np.log(queue)
    mu0 = float(logs.mean())
    sigma0 = float(logs.std())
    if not sigma0 > 0:
        sigma0 = 1.0

    def objectif(p):
        sigma = float(np.exp(np.clip(p[1], -10.0, 10.0)))
        somme = -float(np.sum(_logv_lognormale(queue, xmin, p[0], sigma)))
        return somme if np.isfinite(somme) else _PENALITE

    res = optimize.minimize(objectif, x0=[mu0, np.log(sigma0)], method="Nelder-Mead",
                            options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    if not res.fun < _PENALITE:
        return None
    mu = float(res.x[0])
    sigma = float(np.exp(np.clip(res.x[1], -10.0, 10.0)))
    logv = _logv_lognormale(queue, xmin, mu, sigma)
    if not _ajustement_valide(logv):
        return None
    return AjustementAlternatif("lognormale", {"mu": mu, "sigma": sigma}, logv, 2)


# ---------------------------------------------------------------------------------------
# Weibull (exponentielle etiree) conditionnee
# ---------------------------------------------------------------------------------------
def _logv_weibull(queue, xmin, k, lam):
    """
    ln p(x | x >= xmin) pour une Weibull de forme k et d'echelle lam :
        ln f(x) + (xmin / lam)^k           [car P(X >= xmin) = exp(-(xmin / lam)^k)]
    avec ln f(x) = ln k - ln lam + (k - 1) * ln(x / lam) - (x / lam)^k.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        return (np.log(k) - np.log(lam) + (k - 1.0) * (np.log(queue) - np.log(lam))
                - (queue / lam) ** k + (xmin / lam) ** k)


def ajuster_weibull(queue, xmin):
    """
    Weibull conditionnee a x >= xmin, par maximum de vraisemblance numerique (Nelder-Mead sur
    (ln k, ln lam), depart k = 1, c'est-a-dire une exponentielle). Renvoie None si l'optimisation echoue.
    k < 1 : queue plus lourde que l'exponentielle (exponentielle etiree) ; k = 1 : exponentielle.
    """
    queue = np.asarray(queue, dtype=float)

    def objectif(p):
        k = float(np.exp(np.clip(p[0], -5.0, 5.0)))
        lam = float(np.exp(np.clip(p[1], -20.0, 40.0)))
        somme = -float(np.sum(_logv_weibull(queue, xmin, k, lam)))
        return somme if np.isfinite(somme) else _PENALITE

    echelle0 = float(np.mean(queue))
    res = optimize.minimize(objectif, x0=[0.0, np.log(echelle0)], method="Nelder-Mead",
                            options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    if not res.fun < _PENALITE:
        return None
    k = float(np.exp(np.clip(res.x[0], -5.0, 5.0)))
    lam = float(np.exp(np.clip(res.x[1], -20.0, 40.0)))
    logv = _logv_weibull(queue, xmin, k, lam)
    if not _ajustement_valide(logv):
        return None
    return AjustementAlternatif("weibull", {"k": k, "echelle": lam}, logv, 2)


# ---------------------------------------------------------------------------------------
# Loi de puissance a coupure exponentielle ("tronquee")
# ---------------------------------------------------------------------------------------
def _log_normalisation_puissance_tronquee(alpha, lam, xmin):
    """
    ln Z avec Z = integrale de xmin a l'infini de x^(-alpha) * exp(-lam * x) dx.

    Changement de variable x = xmin + s / lam, puis s = kappa * exp(w) avec kappa = lam * xmin :

        Z = exp(-kappa) / lam * xmin^(-alpha) * I,
        I = integrale en w de  exp( -alpha * ln(1 + exp(w)) - kappa * exp(w) + ln(kappa) + w ) dw.

    L'integrande en w est une bosse lisse de largeur de l'ordre de l'unite, quelle que soit la
    valeur de lam : c'est ce qui permet a l'integration numerique de rester precise meme quand
    lam est tres petit (cas voisin de la loi de puissance pure), la ou une integration directe en x
    manquerait la partie utile. Renvoie nan si l'integrale n'est pas calculable.
    """
    kappa = lam * xmin
    if not (kappa > 0 and np.isfinite(kappa)):
        return float("nan")

    def integrande(w):
        e = np.exp(w)
        return np.exp(-alpha * np.log1p(e) - kappa * e + np.log(kappa) + w)

    bas = -40.0
    haut = float(np.log(50.0 / kappa)) + 5.0        # au-dela, exp(-kappa * e^w) est numeriquement nul
    if haut <= bas:
        return float("nan")
    points = [p for p in (-5.0, 0.0, 5.0) if bas < p < haut]
    valeur, _ = integrate.quad(integrande, bas, haut, points=points or None, limit=200,
                               epsabs=0.0, epsrel=1e-8)
    if not (valeur > 0 and np.isfinite(valeur)):
        return float("nan")
    return float(-kappa - np.log(lam) - alpha * np.log(xmin) + np.log(valeur))


def _logv_puissance_tronquee(queue, xmin, alpha, lam):
    """ln p(x) = -alpha * ln x - lam * x - ln Z, pour chaque point de la queue."""
    return -alpha * np.log(queue) - lam * queue - _log_normalisation_puissance_tronquee(alpha, lam, xmin)


def ajuster_puissance_tronquee(queue, xmin, alpha_depart):
    """
    Loi de puissance a coupure exponentielle p(x) ~ x^(-alpha) * exp(-lam * x), x >= xmin, par maximum
    de vraisemblance numerique (Nelder-Mead sur (alpha, ln lam)).

    Entree :
        alpha_depart : exposant de la loi de puissance pure, utilise comme point de depart avec une
                       coupure tres faible : l'optimisation part donc du modele emboite et ne peut
                       que l'ameliorer. Si la coupure n'apporte rien, lam tend vers 0 et le resultat
                       rejoint la loi de puissance.

    Sortie : AjustementAlternatif ou None si l'optimisation echoue.
    """
    queue = np.asarray(queue, dtype=float)

    def objectif(p):
        alpha = float(p[0])
        lam = float(np.exp(np.clip(p[1], -60.0, 20.0)))
        somme = -float(np.sum(_logv_puissance_tronquee(queue, xmin, alpha, lam)))
        return somme if np.isfinite(somme) else _PENALITE

    alpha0 = float(np.clip(alpha_depart, 1.05, 6.0))
    lam0 = 1.0 / (10.0 * float(queue.max()))
    res = optimize.minimize(objectif, x0=[alpha0, np.log(lam0)], method="Nelder-Mead",
                            options=dict(xatol=1e-6, fatol=1e-8, maxiter=2000))
    if not res.fun < _PENALITE:
        return None
    alpha = float(res.x[0])
    lam = float(np.exp(np.clip(res.x[1], -60.0, 20.0)))
    logv = _logv_puissance_tronquee(queue, xmin, alpha, lam)
    if not _ajustement_valide(logv):
        return None
    return AjustementAlternatif("puissance_tronquee", {"alpha": alpha, "lambda": lam}, logv, 2)


# ---------------------------------------------------------------------------------------
# Ajustement de toutes les alternatives
# ---------------------------------------------------------------------------------------
def ajuster_alternatives(queue, xmin, alpha):
    """
    Ajuste les quatre alternatives sur la queue.

    Entree : queue (x >= xmin), xmin, alpha (exposant de la loi de puissance pure, point de depart
             de la loi tronquee).
    Sortie : dict {nom: AjustementAlternatif}, sans les modeles dont l'ajustement a echoue.
    """
    queue = np.asarray(queue, dtype=float)
    resultats = {
        "exponentielle": ajuster_exponentielle(queue, xmin),
        "lognormale": ajuster_lognormale(queue, xmin),
        "weibull": ajuster_weibull(queue, xmin),
        "puissance_tronquee": ajuster_puissance_tronquee(queue, xmin, alpha),
    }
    return {nom: aj for nom, aj in resultats.items() if aj is not None}


# ---------------------------------------------------------------------------------------
# Comparaisons
# ---------------------------------------------------------------------------------------
def rapport_de_vraisemblance(logv_puissance, alternative):
    """
    Rapport de vraisemblance entre la loi de puissance et une alternative, et sa p-valeur.

        R = somme( logv_puissance - logv_alternative )
        R > 0 : la loi de puissance explique mieux la queue ; R < 0 : c'est l'alternative.

    Test de significativite :
      - alternative "puissance_tronquee" (modele emboite) : chi2 a 1 degre de liberte sur 2|R| ;
      - autres alternatives (non emboitees) : test de Vuong, p = erfc(|R| / (racine(2 n) * sigma)).

    Sortie : (R, p) ; p = nan si la variance des differences est nulle.
    """
    difference = np.asarray(logv_puissance, dtype=float) - alternative.logv_points
    n = len(difference)
    R = float(difference.sum())
    if alternative.nom == "puissance_tronquee":
        return R, float(stats.chi2.sf(2.0 * abs(R), df=1))
    sigma = float(difference.std())
    if not sigma > 0:
        return R, float("nan")
    return R, float(special.erfc(abs(R) / (np.sqrt(2.0 * n) * sigma)))


def verdict_comparaison(R, p, nom_alternative, seuil=LP_SEUIL_P):
    """
    Interprete une comparaison :
      "indecidable"       si p > seuil (ou inconnue) : les donnees ne tranchent pas ;
      "loi_de_puissance"  si p <= seuil et R > 0 ;
      nom_alternative     si p <= seuil et R < 0 (l'alternative est significativement meilleure).
    """
    if not np.isfinite(p) or p > seuil:
        return "indecidable"
    return "loi_de_puissance" if R > 0 else nom_alternative


def poids_aicc(logv_totales, nb_parametres, n):
    """
    Poids d'Akaike (critere corrige AICc) : probabilite relative que chaque modele soit le meilleur.

        AICc = 2k - 2 lnL + 2k(k + 1) / (n - k - 1)      puis poids = exp(-delta / 2), normalises.

    Entree : logv_totales {nom: lnL}, nb_parametres {nom: k}, n (taille de la queue).
    Sortie : {nom: poids} ; des nan si n est trop petit pour la correction.
    """
    aicc = {}
    for nom, lnl in logv_totales.items():
        k = nb_parametres[nom]
        if n - k - 1 <= 0:
            return {nom: float("nan") for nom in logv_totales}
        aicc[nom] = 2.0 * k - 2.0 * lnl + 2.0 * k * (k + 1) / (n - k - 1)
    mini = min(aicc.values())
    brut = {nom: float(np.exp(-0.5 * (a - mini))) for nom, a in aicc.items()}
    total = sum(brut.values())
    return {nom: v / total for nom, v in brut.items()}


def comparer_modeles(queue, xmin, alpha, seuil=LP_SEUIL_P):
    """
    Met la loi de puissance en concurrence avec les alternatives sur la meme queue.

    Entree : queue (x >= xmin), xmin, alpha (loi de puissance ajustee par clauset.py), seuil.

    Sortie : dict
      comparaisons    {alternative: {"R", "p", "favorise"}} -- voir verdict_comparaison()
      poids_aicc      {modele: poids} sur les cinq modeles (loi de puissance comprise)
      meilleur_modele modele de plus grand poids d'Akaike (None si indeterminable)
      parametres      {alternative: parametres ajustes}
    """
    queue = np.asarray(queue, dtype=float)
    n = len(queue)
    logv_pl = logv_loi_de_puissance(queue, xmin, alpha)
    ajustements = ajuster_alternatives(queue, xmin, alpha)

    comparaisons = {}
    logv_totales = {"loi_de_puissance": float(logv_pl.sum())}
    nb_parametres = {"loi_de_puissance": 1}
    for nom, aj in ajustements.items():
        R, p = rapport_de_vraisemblance(logv_pl, aj)
        comparaisons[nom] = dict(R=R, p=p, favorise=verdict_comparaison(R, p, nom, seuil))
        logv_totales[nom] = float(aj.logv_points.sum())
        nb_parametres[nom] = aj.nb_parametres

    poids = poids_aicc(logv_totales, nb_parametres, n)
    valeurs = list(poids.values())
    meilleur = max(poids, key=lambda m: poids[m]) if all(np.isfinite(valeurs)) else None
    return dict(comparaisons=comparaisons, poids_aicc=poids, meilleur_modele=meilleur,
                parametres={nom: aj.parametres for nom, aj in ajustements.items()})


# =======================================================================================
# ===== AJOUT 1 : Adequation de la loi de puissance tronquee (test bootstrap) ===========
# =======================================================================================
# Le critere AICc compare les modeles entre eux (quel est le meilleur ?). Il ne dit pas si
# le modele gagnant est un BON ajustement. Le test bootstrap ci-dessous repond a la question :
# les donnees observees sont-elles compatibles avec la tronquee ajustee ?
#
# Principe (calque sur clauset.p_valeur_adequation) :
#   1. calculer la distance KS entre les donnees et la tronquee ajustee (ks_obs) ;
#   2. simuler B echantillons de la tronquee (alpha_obs, lam_obs), chacun de taille n ;
#   3. reajuster chaque simulation (x_min fixe) et calculer sa distance KS (ks_sim) ;
#   4. p = proportion de simulations avec ks_sim >= ks_obs.
#
# Approximation assumee : x_min est FIXE a sa valeur observee (pas re-choisi a chaque simulation
# comme Clauset). Cout divise par ~3, test valide conditionnellement a x_min.
#
# Lecture : p > 0.1 => tronquee non rejetee (modele plausible) ; p <= 0.1 => tronquee rejetee.
# ---------------------------------------------------------------------------------------

def _simuler_tronquee(n, alpha, lam, xmin, rng, max_iter=100):
    """
    Simule n points d'une loi de puissance tronquee conditionnee a x >= xmin.

    Methode par rejet :
      1. tirer x selon la loi de puissance pure (alpha, xmin) ;
      2. accepter x avec probabilite exp(-lam * (x - xmin)).
    Efficace tant que lam * xmin n'est pas enorme (cas standard : taux d'acceptation > 50 %).

    Si lam <= 0, retourne une loi de puissance pure (cas degenere, mais propre).
    """
    if n <= 0:
        return np.empty(0)
    if lam <= 0:
        # tronquee degeneree : c'est la loi de puissance pure
        return xmin * (1.0 - rng.random(n)) ** (-1.0 / (alpha - 1.0))

    resultats = []
    facteur = 2
    for _ in range(max_iter):
        besoin = n - len(resultats)
        n_cand = max(besoin * facteur, 100)
        # tirage LP pure par inversion de la CDF
        x_cand = xmin * (1.0 - rng.random(n_cand)) ** (-1.0 / (alpha - 1.0))
        # acceptation/rejet
        keep = rng.random(n_cand) < np.exp(-lam * (x_cand - xmin))
        resultats.extend(x_cand[keep].tolist())
        if len(resultats) >= n:
            return np.array(resultats[:n])
        facteur = min(facteur * 2, 100)
    # en pratique jamais atteint pour nos valeurs de lam * xmin
    return np.array(resultats[:n])


def _cdf_tronquee_grille(queue_triee, alpha, lam, xmin, n_points=500):
    """
    CDF conditionnelle F(x | x >= xmin) de la tronquee, evaluee sur une grille
    puis interpolee aux x_i observes. Retourne None si le calcul echoue.

    F(x) = [ ∫_xmin^x t^(-alpha) exp(-lam*t) dt ] / Z
    avec Z = ∫_xmin^inf t^(-alpha) exp(-lam*t) dt   (calculee par _log_normalisation_puissance_tronquee).

    La grille est lineaire de xmin a x_max (le max observe). n_points controlee la precision de
    l'interpolation ; 500 suffit largement pour un test bootstrap avec B=200 (l'erreur
    d'interpolation ~1e-4 est negligeable devant l'incertitude Monte Carlo ~0.035 sur p).
    """
    log_Z = _log_normalisation_puissance_tronquee(alpha, lam, xmin)
    if not np.isfinite(log_Z):
        return None
    Z = np.exp(log_Z)
    if not (Z > 0 and np.isfinite(Z)):
        return None
    x_max = float(queue_triee[-1])
    if x_max <= xmin:
        return None

    # grille : ~500 points entre xmin et x_max
    n_g = max(int(n_points), 200)
    x_grille = np.linspace(xmin, x_max, n_g)

    # integration cumulative de la densite non normalisee
    cdf_grille = np.zeros(n_g)
    cumul = 0.0
    for i in range(1, n_g):
        a, b = x_grille[i - 1], x_grille[i]
        try:
            incr, _ = integrate.quad(
                lambda t: t ** (-alpha) * np.exp(-lam * t),
                a, b, limit=15, epsrel=1e-4)
        except Exception:
            incr = 0.0
        cumul += incr
        cdf_grille[i] = cumul / Z

    # interpolation lineaire aux x_i observes
    return np.interp(queue_triee, x_grille, cdf_grille)


def _distance_ks_conditionnelle(queue_triee, alpha, lam, xmin):
    """
    Distance KS entre la queue observee et la CDF theorique de la tronquee conditionnee.
    Retourne nan si le calcul echoue.
    """
    n = len(queue_triee)
    if n < 2:
        return float("nan")
    cdf = _cdf_tronquee_grille(queue_triee, alpha, lam, xmin)
    if cdf is None:
        return float("nan")
    apres = np.arange(1, n + 1) / n
    avant = np.arange(0, n) / n
    return float(max(np.max(np.abs(apres - cdf)), np.max(np.abs(avant - cdf))))


def p_valeur_adequation_tronquee(queue, alpha_obs, lam_obs, xmin, rng, nb_bootstrap=200):
    """
    p-valeur d'adequation de la loi de puissance tronquee, par bootstrap.

    Entree :
        queue              : observations x >= xmin
        alpha_obs, lam_obs : parametres ajustes sur les donnees observees
        xmin               : seuil (FIXE, pas re-choisi a chaque simulation)
        rng                : numpy.random.Generator
        nb_bootstrap       : nombre de simulations (defaut 200)

    Sortie :
        p-valeur (float) ; nan si aucune simulation n'a pu etre ajustee.

    Lecture : p > 0.1 => tronquee non rejetee (modele plausible)
              p <= 0.1 => tronquee rejetee.

    Cout : O(nb_bootstrap * n * n_g) evaluations de densite, ou n_g est le nombre de points
    de la grille d'interpolation (500 par defaut). Pour sejour (n=5110, B=200), ~30-60s.
    """
    queue = np.asarray(queue, dtype=float)
    queue = queue[queue >= xmin]
    n = len(queue)
    if n < 2 or not (alpha_obs > 1.0 and lam_obs >= 0):
        return float("nan")
    queue_triee = np.sort(queue)

    ks_obs = _distance_ks_conditionnelle(queue_triee, alpha_obs, lam_obs, xmin)
    if not np.isfinite(ks_obs):
        return float("nan")

    superieurs = 0
    valides = 0
    for _ in range(nb_bootstrap):
        sim = _simuler_tronquee(n, alpha_obs, lam_obs, xmin, rng)
        if len(sim) < 2:
            continue
        aj_sim = ajuster_puissance_tronquee(sim, xmin, alpha_depart=alpha_obs)
        if aj_sim is None:
            continue
        valides += 1
        ks_sim = _distance_ks_conditionnelle(
            np.sort(sim),
            aj_sim.parametres["alpha"],
            aj_sim.parametres["lambda"],
            xmin,
        )
        if np.isfinite(ks_sim) and ks_sim >= ks_obs:
            superieurs += 1
    return float(superieurs / valides) if valides > 0 else float("nan")