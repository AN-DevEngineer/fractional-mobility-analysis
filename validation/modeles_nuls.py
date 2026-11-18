# ./validation/modeles_nuls.py
#
# Modeles nuls a MEMOIRE COURTE et test de depassement.
#
# Motivation (Maraun, Rust & Timmer, 2004) : une pente DFA elevee ne prouve pas la memoire longue ;
# il faut la comparer a ce que produirait un processus a memoire courte sur le meme nombre de points.
#
# Modele nul "ar" :
#   1. gaussianise la serie par ses rangs ;
#   2. ajuste un AR(p) (p <= NUL_ORDRE_AR_MAX, choisi par AIC) par Yule-Walker ;
#   3. simule des series AR(p) de meme longueur ;
#   4. leur redonne la marginale EXACTE de la serie observee (echange de rangs, comme l'AAFT).
# Les surrogats ont la meme distribution de valeurs (queues lourdes comprises) et une correlation a
# portee finie : toute pente DFA excedentaire de la serie observee est de la dependance que le modele
# a memoire courte n'explique pas.
#
# CE TEST N'EST PAS UN TEST DE MEMOIRE LONGUE : il dit "non explique par un AR(p)". Des changements
# de regime le declenchent aussi. Sa puissance depend fortement de p et de N (voir
# documentation/RESULTATS_PRELIMINAIRES.md) : presenter p = 1 ET p = 5.

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter
from scipy.special import ndtri
from scipy.stats import rankdata

from config import (N_S, DFA_ECHELLE_MIN, DFA_DIVISEUR_ECHELLE_MAX, DFA_ORDRE, DFA_FENETRE_PENTES_LOCALES,
                    NUL_MODELE, NUL_NB_SURROGATS, NUL_ORDRE_AR_MAX, NUL_ALPHA)
from hurst.dfa import echelles_dfa, fonction_fluctuation, log_fluctuations
from hurst.echelles import pentes_par_ligne


# ---------------------------------------------------------------- ingredients
def gaussianiser(x):
    """Scores normaux des rangs (rang ordinal en cas d'ex aequo)."""
    return ndtri((rankdata(x, method="ordinal") - 0.5) / len(x))


def autocovariance(z, retard_max):
    """Autocovariance empirique (diviseur n) des retards 0..retard_max, par FFT."""
    z = z - z.mean()
    n = len(z)
    taille = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(z, taille)
    return np.fft.irfft(f * np.conj(f), taille)[:retard_max + 1] / n


def ajuster_ar(z, ordre_max):
    """AR(p) de Yule-Walker, p choisi par AIC dans 0..ordre_max. Sortie : (phi, variance d'innovation)."""
    n = len(z)
    g = autocovariance(z, ordre_max)
    meilleur = (np.inf, np.array([]), g[0])
    for p in range(0, ordre_max + 1):
        if p == 0:
            phi, s2 = np.array([]), g[0]
        else:
            phi = solve_toeplitz(g[:p], g[1:p + 1])
            s2 = g[0] - phi @ g[1:p + 1]
        if s2 <= 0:
            continue
        aic = n * np.log(s2) + 2 * p
        if aic < meilleur[0]:
            meilleur = (aic, phi, s2)
    return meilleur[1], float(meilleur[2])


def simuler_ar(phi, s2, n, nb_series, rng, rodage=200):
    """Matrice (nb_series, n) de series AR(p) gaussiennes."""
    eps = np.sqrt(s2) * rng.standard_normal((nb_series, n + rodage))
    return lfilter([1.0], np.concatenate([[1.0], -phi]), eps, axis=1)[:, rodage:]


def imposer_marginale(simulees, x):
    """Remplace, ligne par ligne, les valeurs simulees par celles de x, dans l'ordre des rangs simules."""
    rangs = np.argsort(np.argsort(simulees, axis=1), axis=1)
    return np.sort(x)[rangs]


# ---------------------------------------------------------------- surrogats
def surrogats(x, rng, modele=NUL_MODELE, nb=NUL_NB_SURROGATS, ordre_ar_max=NUL_ORDRE_AR_MAX):
    """Matrice (nb, N) de series du modele nul, de marginale identique a x."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if modele == "melange":
        return np.stack([rng.permutation(x) for _ in range(nb)])
    if modele == "ar":
        phi, s2 = ajuster_ar(gaussianiser(x), ordre_ar_max)
        return imposer_marginale(simuler_ar(phi, s2, n, nb, rng), x)
    raise ValueError(f"modele nul inconnu : {modele!r} (attendu 'ar' ou 'melange')")


# ---------------------------------------------------------------- test
def evaluer_depassement_dfa(x, rng, modele=NUL_MODELE, nb_surrogats=NUL_NB_SURROGATS,
                         ordre_ar_max=NUL_ORDRE_AR_MAX, alpha=NUL_ALPHA,
                         echelle_min=DFA_ECHELLE_MIN, diviseur_max=DFA_DIVISEUR_ECHELLE_MAX,
                         nb_echelles=N_S, ordre=DFA_ORDRE, fenetre=DFA_FENETRE_PENTES_LOCALES,
                         avec_bandes=False):
    """
    Compare la pente DFA globale de x a celles de surrogats a memoire courte.

    Sortie (dict) :
      h_obs         pente DFA observee
      nul_moyenne   moyenne des pentes des surrogats
      nul_seuil     quantile (1 - alpha) des pentes des surrogats
      p             p-valeur unilaterale (H0 = memoire courte), (1 + #{h_surr >= h_obs}) / (nb + 1)
      exces         h_obs - nul_seuil ; > 0 = depassement significatif au seuil alpha
    Avec avec_bandes=True : ajoute les pentes locales observees et leur bande de confiance sous le nul.
    """
    x = np.asarray(x, dtype=float)
    echelles = echelles_dfa(len(x), echelle_min, diviseur_max, nb_echelles)
    nan = float("nan")
    if len(echelles) < 2:
        return dict(h_obs=nan, nul_moyenne=nan, nul_seuil=nan, p=nan, exces=nan)
    log_s = np.log(echelles.astype(float))
    lf_obs = log_fluctuations(fonction_fluctuation(x, echelles, ordre))
    h_obs = float(pentes_par_ligne(log_s, lf_obs)[0])
    S = surrogats(x, rng, modele, nb_surrogats, ordre_ar_max)
    lf_sur = log_fluctuations(fonction_fluctuation(S, echelles, ordre))
    h_sur = pentes_par_ligne(log_s, lf_sur)
    h_sur = h_sur[np.isfinite(h_sur)]
    if len(h_sur) == 0:
        return dict(h_obs=h_obs, nul_moyenne=nan, nul_seuil=nan, p=nan, exces=nan)
    seuil = float(np.quantile(h_sur, 1 - alpha))
    sortie = dict(h_obs=h_obs, nul_moyenne=float(h_sur.mean()), nul_seuil=seuil,
                  p=float((1 + np.sum(h_sur >= h_obs)) / (len(h_sur) + 1)), exces=h_obs - seuil)
    if avec_bandes and len(log_s) >= fenetre:
        def locales(lf):
            return np.stack([pentes_par_ligne(log_s[i:i + fenetre], lf[:, i:i + fenetre])
                             for i in range(len(log_s) - fenetre + 1)], axis=1)
        pentes_sur = locales(lf_sur)
        sortie.update(echelles_locales=np.exp([log_s[i:i + fenetre].mean() for i in range(len(log_s) - fenetre + 1)]),
                      locales_obs=locales(lf_obs)[0],
                      locales_bas=np.nanquantile(pentes_sur, alpha / 2, axis=0),
                      locales_haut=np.nanquantile(pentes_sur, 1 - alpha / 2, axis=0))
    return sortie
