# ./validation/qu.py
#
# Test de Qu (2011) : "A Test Against Spurious Long Memory", Journal of Business & Economic Statistics
# 29(3), 423-438.  (ATTENTION : le memoire cite a tort Econometric Theory 27(2), 283-309.)
#
# HYPOTHESES (a ne pas inverser) :
#   H0 : la serie est stationnaire a memoire longue (VRAIE), f(lambda) ~ G lambda^(-2d), d dans ]-1/2, 1/2[
#   H1 : memoire courte contaminee par des ruptures de moyenne (regimes) ou une tendance lisse
#        (memoire longue "spurieuse")
#   => un REJET signifie "la persistance apparente n'est pas celle d'un processus stationnaire a
#      memoire longue" ; un NON-rejet signifie "un modele a memoire longue pure n'est pas contredit".
#   C'est l'INVERSE de Berkes et al. : les deux tests se completent (voir decision_croisee()).
#
# STATISTIQUE (equation 8 du papier), avec I_j le periodogramme, lambda_j = 2 pi j / n, m ~ n^0.7 :
#   W = sup_{r in [eps, 1]}  (sum_j v_j^2)^(-1/2) | sum_{j <= m r} v_j ( I_j lambda_j^(2 d_chapeau) / G(d_chapeau) - 1 ) |
#   v_j = log j - moyenne(log j),  G(d) = (1/m) sum_j lambda_j^(2d) I_j,  d_chapeau = Whittle local sur m frequences.
# Idee : sous H0 le periodogramme est "uniforme" le long des m frequences basses ; sous H1 il est gonfle
# AUX TOUTES PREMIERES frequences (j << sqrt(n)) seulement, d'ou une derive de la somme partielle.
# Exigence : m / sqrt(n) -> infini (sinon aucune puissance) ; m = n^0.7 recommande par l'auteur.
#
# Loi limite (Theoreme 1) : sup |int_0^r (1+log s) dW - W(1) int_0^r (1+log s) ds - psi(r) int_0^1 (1+log s) dW|,
# psi(r) = int_0^r (1+log s)^2 ds ; valeurs critiques du tableau 1 du papier, re-simulees ici pour verification.
#
# LIMITES A ASSUMER :
#   - le papier annonce un test plutot CONSERVATEUR (taille 1-3 % a 5 % nominal) et peu puissant a n < 1000
#     contre des ruptures rares : un non-rejet n'est PAS une preuve de memoire longue vraie ;
#   - sans preblanchiment le test sur-rejette fortement en presence de forte memoire courte (papier : 35 % pour
#     un ARFIMA(1,0.4,0) avec a1 = 0.8 a n = 500) : le preblanchiment ARFIMA(p,d,q), p,q <= 1 est donc le defaut ;
#   - le papier fait le preblanchiment par vraisemblance gaussienne temporelle (Haslett-Raftery, paquet R fracdiff).
#     Ici il est fait par vraisemblance de WHITTLE (approximation frequentielle) : ecart documente, la taille
#     du test ainsi obtenue est mesuree par simulation (experiences/tests_berkes_qu.py), pas supposee.

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import lfilter

# Tableau 1 du papier : valeurs critiques asymptotiques de W
VALEURS_CRITIQUES = {
    0.02: {0.10: 1.118, 0.05: 1.252, 0.025: 1.374, 0.01: 1.517},
    0.05: {0.10: 1.022, 0.05: 1.155, 0.025: 1.277, 0.01: 1.426},
}


# ------------------------------------------------------------------ ingredients
def periodogramme(x, m):
    """I_j = |sum_t x_t exp(i lambda_j t)|^2 / (2 pi n), j = 1..m, lambda_j = 2 pi j / n."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    f = np.fft.rfft(x - x.mean())
    j = np.arange(1, m + 1)
    lam = 2.0 * np.pi * j / n
    return j, lam, (np.abs(f[1:m + 1]) ** 2) / (2.0 * np.pi * n)


def whittle_local_qu(x, m=None, bornes=(-0.5, 1.0)):
    """
    d_chapeau de Robinson (1995) minimisant R(d) = log G(d) - 2d mean(log lambda_j) sur [bornes] (le papier
    utilise [-1/2, 1] pour eviter l'effet de bord a d proche de 1/2). Sortie : (d, m).
    NB : c'est un estimateur DIFFERENT de hurst.whittle (m = n^0.65, grille H < 1) : ici m = n^0.7 comme Qu.
    """
    n = len(x)
    m = int(n ** 0.7) if m is None else int(m)
    j, lam, I = periodogramme(x, m)
    moy_log = np.mean(np.log(lam))

    def R(d):
        return np.log(np.mean(lam ** (2.0 * d) * I)) - 2.0 * d * moy_log

    grille = np.linspace(bornes[0], bornes[1], 31)
    d0 = grille[int(np.argmin([R(d) for d in grille]))]
    res = minimize_scalar(R, bounds=(max(bornes[0], d0 - 0.06), min(bornes[1], d0 + 0.06)),
                          method="bounded", options=dict(xatol=1e-6))
    return float(res.x), m


def statistique_W(x, m=None, eps=0.02):
    """
    Statistique W de Qu (equation 8). Sortie : (W, d_chapeau, m).
    La somme partielle a r = 1 est nulle par construction (condition du premier ordre de d_chapeau)
    quand d_chapeau est strictement interieur aux bornes.
    """
    x = np.asarray(x, dtype=float)
    d, m = whittle_local_qu(x, m)
    j, lam, I = periodogramme(x, m)
    v = np.log(j) - np.mean(np.log(j))
    g = np.mean(lam ** (2.0 * d) * I)
    u = lam ** (2.0 * d) * I / g - 1.0
    somme = np.cumsum(v * u)
    j_min = max(1, int(np.ceil(eps * m)))
    w = np.max(np.abs(somme[j_min - 1:])) / np.sqrt(np.sum(v ** 2))
    return float(w), d, m


# ------------------------------------------------------------------ preblanchiment ARFIMA(p,d,q), p,q <= 1
def _log_densite_arfima(lam, d, a, b):
    """log g(lambda) de |1 + b e^{-i lam}|^2 / |1 - a e^{-i lam}|^2 * |2 sin(lam/2)|^(-2d)  (variance 1)."""
    z = np.exp(-1j * lam)
    return (np.log(np.abs(1 + b * z) ** 2) - np.log(np.abs(1 - a * z) ** 2) - 2.0 * d * np.log(np.abs(2.0 * np.sin(lam / 2.0))))


def ajuster_arfima_whittle(x, p, q, marge=0.01):
    """
    ARFIMA(p, d, q) avec p, q dans {0, 1} par vraisemblance de Whittle (profilee en la variance), sur toutes
    les frequences de Fourier. Sortie : (a, b, d, aic) avec aic = N log(sigma^2) + 2 (p + q + 1).
    a, b sont bornes a [-1+marge, 1-marge] (stationnarite / inversibilite, comme l'etape 1 du papier).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    f = np.fft.rfft(x - x.mean())
    nj = (n - 1) // 2
    lam = 2.0 * np.pi * np.arange(1, nj + 1) / n
    I = (np.abs(f[1:nj + 1]) ** 2) / (2.0 * np.pi * n)

    def q_whittle(theta):
        d = theta[0]
        a = theta[1] if p == 1 else 0.0
        b = theta[-1] if q == 1 else 0.0
        lg = _log_densite_arfima(lam, d, a, b)
        return np.log(np.mean(I / np.exp(lg))) + np.mean(lg)

    bornes = [(-0.45, 0.45)] + [(-1 + marge, 1 - marge)] * (p + q)
    meilleur = None
    for d_init in (0.0, 0.2):
        x0 = [d_init] + [0.0] * (p + q)
        r = minimize(q_whittle, x0, bounds=bornes, method="L-BFGS-B")
        if meilleur is None or r.fun < meilleur.fun:
            meilleur = r
    th = meilleur.x
    d = float(th[0])
    a = float(th[1]) if p == 1 else 0.0
    b = float(th[-1]) if q == 1 else 0.0
    lg = _log_densite_arfima(lam, d, a, b)
    sigma2 = np.mean(I / np.exp(lg)) * 2.0 * np.pi
    aic = nj * np.log(sigma2) + 2.0 * (p + q + 1)
    return a, b, d, float(aic)


def preblanchir(x):
    """
    Etape 1-2 de la section 5 du papier : choisit ARFIMA(p,d,q), p,q in {0,1}, par AIC, puis filtre
    x*_t = (1 - a L)(1 + b L)^(-1) (x_t - moyenne). La partie FRACTIONNAIRE n'est pas retiree : seul le
    court terme l'est. Sortie : (x_etoile, (p, q, a, b, d)).
    """
    x = np.asarray(x, dtype=float)
    meilleur = None
    for p in (0, 1):
        for q in (0, 1):
            a, b, d, aic = ajuster_arfima_whittle(x, p, q)
            if meilleur is None or aic < meilleur[0]:
                meilleur = (aic, p, q, a, b, d)
    _, p, q, a, b, d = meilleur
    xc = x - x.mean()
    x_etoile = lfilter([1.0, -a], [1.0, b], xc)
    return x_etoile, (p, q, a, b, d)


# ------------------------------------------------------------------ test
def valeur_critique_W(alpha=0.05, eps=0.02):
    """Valeur critique asymptotique du tableau 1 du papier."""
    if eps not in VALEURS_CRITIQUES or alpha not in VALEURS_CRITIQUES[eps]:
        raise ValueError(f"valeur critique tabulee seulement pour eps in {list(VALEURS_CRITIQUES)} et "
                         f"alpha in {list(VALEURS_CRITIQUES[0.02])} ; ou utiliser simuler_loi_limite()")
    return VALEURS_CRITIQUES[eps][alpha]


def test_qu(x, eps=0.02, alpha=0.05, m=None, avec_preblanchiment=True):
    """
    Test de Qu. H0 = vraie memoire longue stationnaire ; rejet = memoire longue spurieuse.
    Sortie (dict) : rejette (True = rejet de H0), w, seuil, d (Whittle local), m, modele_prewhitening (p,q,a,b,d).
    Serie trop courte (n < 128) : non concluant (rejette = None), car m = n^0.7 fournirait moins de ~25 frequences
    et le trimming eps * m devient degenere.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 128:
        return dict(rejette=None, w=float("nan"), seuil=valeur_critique_W(alpha, eps), d=float("nan"),
                    m=int(n ** 0.7), modele_prewhitening=None)
    modele = None
    y = x
    if avec_preblanchiment:
        y, modele = preblanchir(x)
    w, d, mm = statistique_W(y, m, eps)
    seuil = valeur_critique_W(alpha, eps)
    return dict(rejette=bool(w > seuil), w=w, seuil=seuil, d=d, m=mm, modele_prewhitening=modele)


def simuler_loi_limite(nb=4000, n_grille=4000, eps=0.02, rng=None):
    """
    Simule la loi limite du Theoreme 1 (papier : 10 000 repetitions, grille 10 000) pour verifier le tableau 1.
    Sortie : tableau des `nb` realisations de sup_{r in [eps,1]} |...|.
    """
    rng = np.random.default_rng(20261018) if rng is None else rng
    s = (np.arange(1, n_grille + 1) - 0.5) / n_grille
    f = 1.0 + np.log(s)                                   # (1 + log s)
    F = np.cumsum(f) / n_grille                           # int_0^r (1 + log s) ds
    psi = np.cumsum(f ** 2) / n_grille                    # int_0^r (1 + log s)^2 ds   (psi(1) = 1)
    r0 = int(np.ceil(eps * n_grille))
    sortie = np.empty(nb)
    lot = 200
    for debut in range(0, nb, lot):
        k = min(lot, nb - debut)
        dW = rng.standard_normal((k, n_grille)) / np.sqrt(n_grille)
        I = np.cumsum(f * dW, axis=1)                     # int_0^r (1 + log s) dW(s)
        W1 = dW.sum(axis=1, keepdims=True)
        proc = I - W1 * F - psi * I[:, -1:]
        sortie[debut:debut + k] = np.max(np.abs(proc[:, r0 - 1:]), axis=1)
    return sortie


# ------------------------------------------------------------------ lecture croisee des deux tests
def decision_croisee(rejet_berkes, rejet_qu):
    """
    Table de lecture des deux tests, SANS surinterpretation :
      Berkes rejette (memoire courte + rupture(s) rejetee) ET Qu ne rejette pas : compatible avec une memoire longue vraie ;
      Berkes ne rejette pas ET Qu rejette : compatible avec memoire courte + changements de regime (persistance spurieuse) ;
      les deux rejettent : ni "memoire courte + ruptures" ni "memoire longue pure" ne suffisent (processus mixte,
                           memoire longue + rupture, ou non-stationnarite plus riche) ;
      aucun ne rejette : le test ne tranche pas (faible puissance a n ~ 10^2-10^3).
    """
    if rejet_berkes is None or rejet_qu is None:
        return "non concluant"
    if rejet_berkes and not rejet_qu:
        return "memoire longue compatible"
    if (not rejet_berkes) and rejet_qu:
        return "memoire courte + regimes compatible"
    if rejet_berkes and rejet_qu:
        return "ni l'un ni l'autre (processus mixte)"
    return "indetermine (aucun rejet)"
