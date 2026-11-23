# ./profilage/caracteristiques.py
#
# Vecteur de caracteristiques fractionnaires du chapitre 4 (equation 4.6) :
#     v = (H_DFA, H_RS, H_Whittle, alpha, S_alpha, D_f)  dans R^6
#
# DEFINITIONS OPERATIONNELLES (le texte du memoire laisse des choix ouverts : ils sont fixes ICI, a justifier) :
#   * H_DFA, H_RS (corrige, Anis-Lloyd), H_Whittle (local) : les estimateurs du depot (hurst/), sur la serie d_i.
#   * alpha (section 4.5) : "minimisation de l'erreur de prediction a un pas de la serie filtree par D^alpha sur la
#     grille {0.1, ..., 0.9}". Le texte ne dit pas QUEL predicteur. Choix retenu : y = D^alpha x (Grunwald-Letnikov,
#     K termes), puis predicteur lineaire a un pas y_t ~ phi y_{t-1} ; erreur de prediction NORMALISEE
#     1 - rho_1(y)^2 (sans normalisation, l'erreur depend de l'echelle de y, donc de alpha, et la grille n'est pas comparable).
#     alpha_chapeau = argmin. Consequence a connaitre : maximise la predictibilite lineaire a un pas de la serie filtree.
#   * S_alpha (4.7) : entropie de Renyi d'ordre `ordre_renyi` (defaut 2) de la distribution empirique des
#     INCREMENTS de vitesse (histogramme, regle de Freedman-Diaconis, 8 a 64 classes), NORMALISEE par log(nb classes)
#     pour rester comparable d'une fenetre a l'autre (le texte ne normalise pas : choix a declarer).
#   * D_f : dimension de boites de la trajectoire projetee en metres (equirectangulaire), echelles 2^-k, k = 2..6.
#   NB : la section 4.6 (spectre fractionnaire / FrFT, energie en 3 bandes) decrit une caracteristique qui N'EST PAS
#   dans le vecteur (4.6) ; elle n'est pas implementee ici.

import numpy as np

from hurst.dfa import dfa_complet
from hurst.rs import rs_complet
from hurst.whittle import whittle_local

NOMS = ["H_DFA", "H_RS", "H_Whittle", "alpha", "S_renyi", "D_f"]


# ------------------------------------------------------------------ derivee de Grunwald-Letnikov
def poids_gl(alpha, K):
    """w_0 = 1, w_k = w_{k-1} (k - 1 - alpha) / k  (equation 4.2) ; |w_k| ~ k^-(1+alpha)."""
    w = np.empty(K + 1)
    w[0] = 1.0
    for k in range(1, K + 1):
        w[k] = w[k - 1] * (k - 1 - alpha) / k
    return w


def derivee_gl(x, alpha, K=10):
    """(D^alpha x)_t = sum_{k=0}^{min(K, t)} w_k x_{t-k} ; convolution causale, zeros avant le debut."""
    x = np.asarray(x, dtype=float)
    return np.convolve(x, poids_gl(alpha, K))[:len(x)]


def ordre_fractionnaire(x, K=10, grille=None):
    """alpha_chapeau (voir l'en-tete du module). Renvoie (alpha, erreurs_normalisees_par_alpha)."""
    grille = np.round(np.arange(0.1, 0.91, 0.1), 2) if grille is None else np.asarray(grille)
    x = np.asarray(x, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    erreurs = []
    for a in grille:
        y = derivee_gl(x, a, K)[K:]                  # on jette les K premiers points (effet de bord)
        if len(y) < 5 or y.std() < 1e-12:
            erreurs.append(np.inf)
            continue
        r1 = np.corrcoef(y[:-1], y[1:])[0, 1]
        erreurs.append(1.0 - r1 ** 2)
    erreurs = np.array(erreurs)
    return float(grille[int(np.argmin(erreurs))]), erreurs


# ------------------------------------------------------------------ entropie de Renyi et dimension de boites
def entropie_renyi(p, ordre=2.0):
    """S_a(p) = ln(sum p_i^a) / (1 - a) ; Shannon pour a = 1 (equation 4.5)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    p = p / p.sum()
    if abs(ordre - 1.0) < 1e-9:
        return float(-np.sum(p * np.log(p)))
    return float(np.log(np.sum(p ** ordre)) / (1.0 - ordre))


def entropie_increments_vitesse(v, ordre=2.0):
    """Entropie de Renyi NORMALISEE (par log du nb de classes) des increments de vitesse."""
    dv = np.diff(np.asarray(v, dtype=float))
    dv = dv[np.isfinite(dv)]
    if len(dv) < 10 or np.ptp(dv) == 0:
        return float("nan")
    bords = np.histogram_bin_edges(dv, bins="fd")
    nb = int(np.clip(len(bords) - 1, 8, 64))
    h, _ = np.histogram(dv, bins=nb)
    return entropie_renyi(h / h.sum(), ordre) / np.log(nb)


def dimension_boites(lat, lon, ks=range(2, 7)):
    """
    Dimension de boites de la trajectoire (projection equirectangulaire en metres, boite englobante carree
    normalisee a 1). Pente de log N(eps) contre log(1/eps), eps = 2^-k. Trajectoire ponctuelle : nan.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = (lon - lon.mean()) * 111320.0 * np.cos(np.deg2rad(lat.mean()))
    y = (lat - lat.mean()) * 110574.0
    cote = max(np.ptp(x), np.ptp(y))
    if cote < 1e-6:
        return float("nan")
    u = (x - x.min()) / cote
    w = (y - y.min()) / cote
    logs_inv_eps, logs_n = [], []
    for k in ks:
        n_cases = 2 ** k
        i = np.minimum((u * n_cases).astype(int), n_cases - 1)
        j = np.minimum((w * n_cases).astype(int), n_cases - 1)
        logs_inv_eps.append(k * np.log(2.0))
        logs_n.append(np.log(len(np.unique(i * n_cases + j))))
    return float(np.polyfit(logs_inv_eps, logs_n, 1)[0])


# ------------------------------------------------------------------ vecteur
def vecteur_caracteristiques(d, v, lat, lon, ordre_renyi=2.0):
    """
    Vecteur v in R^6 (NOMS) d'un morceau de trajectoire.
      d : distances inter-points (m) ; v : vitesses (km/h) ; lat, lon : positions (len(d) + 1 points).
    Toute caracteristique non calculable vaut nan (le NaN n'est jamais remplace silencieusement).
    """
    d = np.asarray(d, dtype=float)
    h_dfa = dfa_complet(d).h
    h_rs = rs_complet(d).h_corrige
    h_wh = whittle_local(d).h
    alpha, _ = ordre_fractionnaire(d)
    s = entropie_increments_vitesse(v, ordre_renyi)
    df = dimension_boites(lat, lon)
    return np.array([h_dfa, h_rs, h_wh, alpha, s, df], dtype=float)


def sequence_de_fenetres(d, v, lat, lon, taille=128, pas=64, t_max=None):
    """
    Sequence (T, 6) de vecteurs calcules sur des FENETRES glissantes de `taille` points (pas `pas`).
    C'est la definition retenue de la sequence x_{1:T} de la section 5.7.1 (le texte ne dit pas de quoi la sequence
    est faite). Contrainte : un H n'est pas estimable sur moins de ~100 points, donc T = 1 + (n - taille) // pas
    (~10 pour un segment median de 721 points) ; t_max tronque (on garde les t_max premieres fenetres).
    Sortie : tableau (T, 6), possiblement avec des nan.
    """
    n = len(d)
    debuts = list(range(0, n - taille + 1, pas))
    if t_max is not None:
        debuts = debuts[:t_max]
    lignes = [vecteur_caracteristiques(d[a:a + taille], v[a:a + taille], lat[a:a + taille + 1], lon[a:a + taille + 1])
              for a in debuts]
    return np.array(lignes) if lignes else np.empty((0, 6))
