# ./validation/berkes.py
#
# Test de Berkes, Horvath, Kokoszka & Shao (2006) : "On discriminating between long-range dependence
# and changes in mean", The Annals of Statistics 34(3), 1140-1165. (arXiv:math/0607803)
#
# HYPOTHESES (a ne pas inverser) :
#   H0 : memoire COURTE avec au plus u changements de moyenne (modele (2.1) du papier)
#   HA : memoire LONGUE stationnaire (H dans ]1/2, 1[)
#   => un REJET de H0 est un argument EN FAVEUR de la memoire longue ; un NON-rejet dit que la
#      persistance observee est compatible avec une memoire courte + rupture(s) de moyenne.
#
# PRINCIPE (section 2 du papier) :
#   1. statistique CUSUM sur toute la serie -> estimateur k_chapeau de la rupture (2.11), qu'il y ait
#      une rupture ou non ;
#   2. on coupe en k_chapeau ; sur CHAQUE morceau on recalcule un CUSUM normalise par SON PROPRE
#      estimateur de variance a long terme (Bartlett) : T_{n,1}, T_{n,2} (2.12-2.13) ;
#   3. M_n = max(T_{n,1}, T_{n,2}) (2.17). Sous H0, M_n -> max de deux sup|pont brownien| independants.
#      Sous HA, M_n -> infini : car l'estimateur de Bartlett d'une serie a memoire longue croit comme
#      q^(2H-1) et non comme une constante, donc la normalisation "sous-estime" le CUSUM.
#   Pour plusieurs ruptures : segmentation binaire (Vostrikova 1981), section 3 du papier.
#
# POINTS D'ATTENTION POUR LA SOUTENANCE :
#   - la theorie impose q(n) -> infini et q(n) (log n)^4 = O(n) : a n ~ 700 points (log n)^4 ~ 1800 > n,
#     la condition asymptotique n'a aucun contenu. Le choix de q est donc une decision empirique :
#     on RAPPORTE la sensibilite a q (cf. experiences/tests_berkes_qu.py) au lieu de la cacher.
#   - le test suppose que les ruptures sont GRANDES (n * Delta^2 -> infini) : de petites derives
#     de moyenne ne sont pas couvertes.
#   - le nombre maximal de ruptures u est un REGLAGE : on rapporte u = 1, 2 et 3.

import numpy as np
from scipy.optimize import brentq


# ------------------------------------------------------------------ loi limite
def kolmogorov_cdf(x, termes=200):
    """P(sup |pont brownien| <= x) = 1 - 2 sum_{k>=1} (-1)^(k-1) exp(-2 k^2 x^2)."""
    x = np.asarray(x, dtype=float)
    k = np.arange(1, termes + 1)[:, None]
    s = 1.0 - 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k ** 2 * np.atleast_1d(x) ** 2), axis=0)
    s = np.where(np.atleast_1d(x) <= 0, 0.0, s)
    return float(s[0]) if s.shape == (1,) else s


def valeur_critique(alpha, u=1):
    """
    c(u) du papier (section 3) : P(max de u sup|ponts browniens| independants > c(u)) = alpha,
    soit F(c)^u = 1 - alpha avec F la fonction de repartition de Kolmogorov.
    Verification : u = 2 -> 1.36 / 1.48 / 1.72 pour alpha = 10 % / 5 % / 1 % (valeurs du papier).
    """
    cible = (1.0 - alpha) ** (1.0 / u)
    return float(brentq(lambda c: kolmogorov_cdf(c) - cible, 0.3, 6.0))


# ------------------------------------------------------------------ ingredients
def largeur_bartlett(n, regle="nw"):
    """
    Largeur de bande q(n) de l'estimateur de Bartlett.
      "nw"    : Newey-West, floor(4 (n/100)^(2/9))       (defaut ; petite, adaptee aux n ~ 10^2-10^3)
      "cube"  : floor(n^(1/3))
      "log15" : floor(15 log10(n))                       (choix du papier, section 3, n = 2021)
      entier  : valeur imposee
    """
    if isinstance(regle, (int, np.integer)):
        q = int(regle)
    elif regle == "nw":
        q = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    elif regle == "cube":
        q = int(np.floor(n ** (1.0 / 3.0)))
    elif regle == "log15":
        q = int(np.floor(15.0 * np.log10(max(n, 2))))
    else:
        raise ValueError(f"regle de largeur inconnue : {regle!r}")
    return max(1, min(q, n - 2))


def variance_long_terme(x, q):
    """Estimateur de Bartlett s^2 = g0 + 2 sum_{j<=q} (1 - j/(q+1)) g_j (formules 2.15-2.16, 3.2)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    z = x - x.mean()
    taille = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(z, taille)
    g = np.fft.irfft(f * np.conj(f), taille)[:q + 1] / n          # diviseur n, comme (3.3)
    poids = 1.0 - np.arange(1, q + 1) / (q + 1.0)
    return float(g[0] + 2.0 * np.sum(poids * g[1:q + 1]))


def cusum(x):
    """
    Pont des sommes partielles B_k = S_k - (k/n) S_n, k = 1..n.
    Sortie : (max |B_k|, k_chapeau) ; k_chapeau = PREMIER indice realisant le max, comme (2.11).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = np.cumsum(x)
    pont = s - (np.arange(1, n + 1) / n) * s[-1]
    a = np.abs(pont)
    k = int(np.argmax(a)) + 1
    return float(a[k - 1]), k


def statistique_T(x, regle_q="nw", longueur_min=20):
    """
    T = max_k |S_k - (k/n) S_n| / (sqrt(n) s_n)  (3.1) ; renvoie (T, k_chapeau).
    NaN si le morceau est trop court ou de variance a long terme non positive (Bartlett peut etre < 0
    a taille finie avec un noyau tronque : on le signale par NaN plutot que de le masquer).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < longueur_min:
        return float("nan"), None
    m, k = cusum(x)
    s2 = variance_long_terme(x, largeur_bartlett(n, regle_q))
    if not np.isfinite(s2) or s2 <= 0:
        return float("nan"), k
    return float(m / (np.sqrt(n) * np.sqrt(s2))), k


# ------------------------------------------------------------------ test
def test_berkes(x, u_max=1, alpha=0.05, regle_q="nw", longueur_min=20):
    """
    Test de Berkes et al. avec au plus u_max ruptures de moyenne sous H0 (segmentation binaire).

    Algorithme (section 3 du papier). On part d'un seul morceau. A l'etape u (u morceaux) :
      - M_u = max des T des morceaux ; si M_u <= c(u) : on ne rejette pas H0 (memoire courte + (u-1) ruptures) ;
      - sinon, si u - 1 = u_max : on REJETTE H0 (memoire longue) ;
      - sinon on coupe le morceau de plus grand T en son k_chapeau et on recommence.
    Un morceau trop court ou degenere (T = NaN) rend le test non concluant (verdict = None).

    Sortie (dict) :
      rejette   True = rejet de H0 (argument pour la memoire longue) ; False = non rejet ; None = non concluant
      m_stat    derniere valeur de M_u calculee ;  seuil   c(u) correspondant
      n_ruptures_estimees  nombre de ruptures retenues sous H0 quand H0 n'est pas rejetee
      points_de_rupture    indices des ruptures estimees
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    bornes = [0, n]
    resultat = dict(rejette=None, m_stat=float("nan"), seuil=float("nan"), n_ruptures_estimees=None,
                    points_de_rupture=[], n=n)
    for u in range(1, u_max + 2):
        morceaux = [(bornes[i], bornes[i + 1]) for i in range(len(bornes) - 1)]
        stats = [statistique_T(x[a:b], regle_q, longueur_min) for a, b in morceaux]
        t_vals = np.array([s[0] for s in stats])
        if not np.all(np.isfinite(t_vals)):
            return resultat                      # non concluant
        m_stat = float(t_vals.max())
        seuil = valeur_critique(alpha, u)
        resultat.update(m_stat=m_stat, seuil=seuil)
        if m_stat <= seuil:
            resultat.update(rejette=False, n_ruptures_estimees=u - 1, points_de_rupture=sorted(bornes[1:-1]))
            return resultat
        if u - 1 == u_max:
            resultat.update(rejette=True, n_ruptures_estimees=None, points_de_rupture=sorted(bornes[1:-1]))
            return resultat
        i = int(np.argmax(t_vals))
        a, b = morceaux[i]
        k = stats[i][1]
        bornes = sorted(set(bornes + [a + k]))
        if len(bornes) - 1 == u:                 # la coupure n'a rien ajoute (k au bord) : on arrete
            resultat.update(rejette=True, points_de_rupture=sorted(bornes[1:-1]))
            return resultat
    return resultat
