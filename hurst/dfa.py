# ./hurst/dfa.py
#
# Methode DFA (Detrended Fluctuation Analysis) pour l'estimation de H
#
# Fusionne dfa/fluctuation.py et dfa/hurst.py en un seul fichier.
# Le module commun (extraction_distances, profil_cumule) vient de hurst/serie_distances.py.
#
# Reference : Peng, C.K. et al. (1994). Mosaic organization of DNA nucleotides.
#             Physical Review E, 49(2), 1685.

from math import log, exp, floor, sqrt

from hurst.serie_distances import extraction_distances, profil_cumule
from config import N_S


# ---------------------------------------------------------------
# Generation des tailles de fenetres s
# ---------------------------------------------------------------

def generer_valeurs_s(N, n_s):
    """
    Genere les tailles de fenetres s pour la methode DFA

    Les valeurs sont espacees exponentiellement entre s_min et s_max
    pour obtenir des points regulierement repartis en echelle log-log.

    Formule pour le j-ieme point (j = 1, ..., n_s) :
        log(s_j) = log(s_min) + ((j-1) / (n_s-1)) * (log(s_max) - log(s_min))
        s_j = floor(exp(log(s_j)))

    Contraintes :
        s_min = 4        (minimum pour detrending lineaire : 2*degre + 2)
        s_max = N // 4   (garantit au moins 4 fenetres completes)

    Entree :
        N   : longueur de la serie (nombre de distances)
        n_s : nombre de valeurs a generer avant suppression des doublons

    Sortie :
        liste d'entiers : valeurs distinctes de s en ordre croissant
        liste vide si N trop court
    """
    s_min = 4
    s_max = N // 4

    # Garde : si s_max <= s_min, la serie est trop courte
    if s_max <= s_min:
        return []

    ensemble_s = []

    for j in range(1, n_s + 1):
        log_sj = log(s_min) + ((j - 1) / (n_s - 1)) * (log(s_max) - log(s_min))
        sj     = floor(exp(log_sj))
        ensemble_s.append(sj)

    # Suppression des doublons (valeurs identiques apres floor)
    ensemble_s_unique = []
    for s in ensemble_s:
        if s not in ensemble_s_unique:
            ensemble_s_unique.append(s)

    return ensemble_s_unique


# ---------------------------------------------------------------
# Calcul des fluctuations
# ---------------------------------------------------------------

def variance_locale(s, sum_k, sum_k2, Y, indice_debut):
    """
    Calcule la variance locale des residus dans une fenetre

    On ajuste une tendance lineaire (y = a*k + b) sur la fenetre
    par moindres carres, puis on calcule la variance des residus.

    Les sommes sum_k et sum_k2 sont precalculees en dehors de cette
    fonction pour eviter de les recalculer pour chaque fenetre.

        sum_k  = sum_{k=0}^{s-1} k  = s*(s-1)/2
        sum_k2 = sum_{k=0}^{s-1} k^2 = s*(s-1)*(2s-1)/6

    Entree :
        s           : taille de la fenetre
        sum_k       : sum_{k=0}^{s-1} k  (precalcule)
        sum_k2      : sum_{k=0}^{s-1} k^2 (precalcule)
        Y           : profil cumule complet
        indice_debut: indice de debut de la fenetre dans Y

    Sortie :
        float : variance locale des residus
    """
    sum_yk   = 0
    sum_k_yk = 0

    for i in range(indice_debut, indice_debut + s):
        k         = i - indice_debut
        sum_yk   += Y[i]
        sum_k_yk += k * Y[i]

    # Coefficients de la droite de regression : Y ~ a*k + b
    denominateur = s * sum_k2 - sum_k * sum_k
    a = (s * sum_k_yk - sum_yk * sum_k) / denominateur
    b = (sum_yk - a * sum_k) / s

    # Variance des residus
    variance = 0
    for i in range(indice_debut, indice_debut + s):
        k       = i - indice_debut
        residu  = Y[i] - (a * k + b)
        variance += residu * residu

    return variance / s


def fluctuation(s, Y):
    """
    Calcule F(s) pour une taille de fenetre s

    F(s) = sqrt( (1/M) * sum_{v=1}^{M} F^2(s, v) )

    ou M est le nombre de fenetres non chevauchantes de taille s
    et F^2(s, v) est la variance locale dans la v-ieme fenetre.

    Entree :
        s : taille de la fenetre
        Y : profil cumule complet

    Sortie :
        float : F(s)
        None  : si aucune fenetre n'est disponible
    """
    taille = len(Y)

    # Precalcul des sommes pour la regression (constant pour toutes
    # les fenetres de meme taille s -> calcul une seule fois)
    sum_k  = (s * (s - 1)) // 2
    sum_k2 = (s * (s - 1) * (2 * s - 1)) // 6

    somme  = 0
    nb_fen = 0   # nombre de fenetres effectivement parcourues
    i      = 0

    while (i + s) <= taille:
        somme  += variance_locale(s, sum_k, sum_k2, Y, i)
        nb_fen += 1
        i      += s

    if nb_fen == 0:
        return None

    return sqrt(somme / nb_fen)


def paires_s_f(ensemble_s, Y):
    """
    Calcule les paires (s, F(s)) pour chaque valeur de s

    Filtre les paires invalides (F = None ou = 0)
    pour eviter log(0) dans la regression.

    Entree :
        ensemble_s : liste des valeurs de s
        Y          : profil cumule complet

    Sortie :
        liste de tuples (s, F(s))
    """
    paires = []

    for s in ensemble_s:
        f = fluctuation(s, Y)

        # log(0) interdit et log(None) plante -> on filtre
        if f is not None and f > 0:
            paires.append((s, f))

    return paires


# ---------------------------------------------------------------
# Regression log-log et estimation de H
# ---------------------------------------------------------------

def estimer_H_regression(paires):
    """
    Estime le parametre de Hurst H par regression lineaire log-log

    Soit U_j = log(s_j) et V_j = log(F(s_j)).
    On cherche la pente H de la droite V = H*U + C par moindres carres :

        H = (n * sum(Uj*Vj) - sum(Uj) * sum(Vj)) /
            (n * sum(Uj^2)  - sum(Uj)^2)

    Entree :
        paires : liste de tuples (s, F(s))

    Sortie :
        float : estimation de H
    """
    n          = len(paires)
    somme_uj   = 0
    somme_vj   = 0
    somme_ujvj = 0
    somme_uj2  = 0

    for i in range(n):
        uj = log(paires[i][0])
        vj = log(paires[i][1])

        somme_uj   += uj
        somme_vj   += vj
        somme_ujvj += uj * vj
        somme_uj2  += uj * uj

    H = (n * somme_ujvj - somme_uj * somme_vj) / \
        (n * somme_uj2  - somme_uj * somme_uj)

    return H


# ---------------------------------------------------------------
# Point d'entree principal
# ---------------------------------------------------------------

def DFA(segment):
    """
    Applique la methode DFA complete sur un segment et retourne H

    Accepte deux formats (detection automatique) :
        - Liste de points GPS [[lat, lon, ...], ...]
        - Serie de floats [d1, d2, ...] (fBm synthetique)

    Pipeline :
        1. Extraire les distances (ou utiliser la serie directe)
        2. Construire le profil cumule Y
        3. Generer les tailles de fenetres s (log-espacees)
        4. Calculer les paires (s, F(s))
        5. Estimer H par regression log-log

    Entree :
        segment : liste de points GPS ou liste de float

    Sortie :
        float : estimation de H
        None  : si le segment est trop court ou l'estimation echoue
    """
    distances = extraction_distances(segment)
    N         = len(distances)

    # Minimum absolu pour la DFA : s_min=4, s_max=N//4 >= 5 -> N >= 20
    # En pratique MIN_POINTS=200 dans config.py garantit largement ce seuil
    if N < 16:
        print(f"[AVERTISSEMENT] Segment trop court pour DFA : N={N}")
        return None

    Y          = profil_cumule(distances)
    ensemble_s = generer_valeurs_s(N, N_S)

    if len(ensemble_s) < 2:
        print(f"[AVERTISSEMENT] Pas assez de valeurs de s pour DFA : N={N}")
        return None

    paires = paires_s_f(ensemble_s, Y)

    if len(paires) < 2:
        print(f"[AVERTISSEMENT] Pas assez de paires valides pour la regression DFA")
        return None

    H = estimer_H_regression(paires)

    return H


# ============================================================================
# VERSION RAPIDE (V5)
# ----------------------------------------------------------------------------
# Meme calcul que DFA() ci-dessus (qui reste la REFERENCE LISIBLE, ecrite a la
# main), mais vectorise avec numpy : ~100 fois plus rapide, et capable de
# traiter un lot de series d'un coup (surrogats du modele nul).
# tests/test_dfa.py verifie que les deux donnent les memes valeurs.
#
# Lecture de l'exposant (confusion frequente) :
#   - bruit gaussien fractionnaire (fGn, stationnaire)            -> alpha = H
#   - mouvement brownien fractionnaire (fBm, non stationnaire)    -> alpha = H + 1
#   - alpha > 1 sur des donnees signale donc une serie hors du domaine du fGn.
# Une pente globale ajustee sur toute la courbe ne prouve pas une loi d'echelle
# (Maraun, Rust & Timmer, 2004) : on expose aussi les pentes locales et les
# pentes des moities basse / haute des echelles.
# ============================================================================
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from config import (DFA_ECHELLE_MIN, DFA_DIVISEUR_ECHELLE_MAX, DFA_ORDRE, DFA_FENETRE_PENTES_LOCALES)
from hurst.echelles import grille_echelles, pente_mco, pentes_par_ligne, pentes_glissantes


@lru_cache(maxsize=512)
def base_polynomiale(s, ordre):
    """Base orthonormee du sous-espace des polynomes de degre <= ordre sur s points."""
    k = np.arange(s, dtype=float)
    k = (k - k.mean()) / max(s - 1, 1)      # centrage et mise a l'echelle : meilleur conditionnement
    q, _ = np.linalg.qr(np.vander(k, ordre + 1, increasing=True))
    return q


def echelles_dfa(n, echelle_min=DFA_ECHELLE_MIN, diviseur_max=DFA_DIVISEUR_ECHELLE_MAX, nb_echelles=N_S):
    """Echelles utilisees pour une serie de longueur n (vide si trop courte)."""
    if n < 16:
        return np.array([], dtype=int)
    return grille_echelles(echelle_min, n // diviseur_max, nb_echelles)


def fonction_fluctuation(X, echelles, ordre=DFA_ORDRE):
    """
    F(s) pour une serie (1D) ou un lot de series (2D, une par ligne). NaN si non calculable.

    Profil cumule des ecarts a la moyenne ; fenetres disjointes de taille s ; retrait d'un
    polynome d'ordre `ordre` dans chaque fenetre ; F(s) = racine de la moyenne des variances
    residuelles (comme variance_locale / fluctuation ci-dessus).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    nb_series, n = X.shape
    Y = np.cumsum(X - X.mean(axis=1, keepdims=True), axis=1)
    F = np.full((nb_series, len(echelles)), np.nan)
    for j, s in enumerate(echelles):
        s = int(s)
        nb_fen = n // s
        if nb_fen < 1 or s <= ordre + 1:
            continue
        W = Y[:, :nb_fen * s].reshape(nb_series, nb_fen, s)
        proj = W @ base_polynomiale(s, ordre)
        ssr = np.maximum((W * W).sum(-1) - (proj * proj).sum(-1), 0.0)   # somme des carres residuels
        F[:, j] = np.sqrt(ssr.mean(axis=1) / s)
    return F


def log_fluctuations(F):
    """log F, avec NaN la ou F <= 0 ou non calculable."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.where(F > 0, F, np.nan))


@dataclass(frozen=True)
class ResultatDFA:
    n: int
    echelles: np.ndarray
    fluctuations: np.ndarray
    h: float                    # pente globale (exposant DFA alpha)
    ordonnee: float
    echelles_locales: np.ndarray  # echelle (geometrique) au centre de chaque fenetre de pente locale
    pentes_locales: np.ndarray
    h_bas: float                # pente sur la moitie basse des echelles
    h_haut: float               # pente sur la moitie haute des echelles

    @property
    def ecart_pentes(self):
        """h_haut - h_bas : 0 = loi d'echelle unique ; grand = crossover (deux regimes d'echelle)."""
        return self.h_haut - self.h_bas


def dfa_complet(serie, echelle_min=DFA_ECHELLE_MIN, diviseur_max=DFA_DIVISEUR_ECHELLE_MAX,
                nb_echelles=N_S, ordre=DFA_ORDRE, fenetre=DFA_FENETRE_PENTES_LOCALES):
    """DFA complete d'une serie : pente globale, pentes locales, pentes des moities. h = nan si trop courte."""
    x = np.asarray(serie, dtype=float)
    n = len(x)
    echelles = echelles_dfa(n, echelle_min, diviseur_max, nb_echelles)
    vide = np.array([])
    if len(echelles) < 2:
        return ResultatDFA(n, echelles, vide, np.nan, np.nan, vide, vide, np.nan, np.nan)
    F = fonction_fluctuation(x, echelles, ordre)[0]
    ok = np.isfinite(F) & (F > 0)
    ls, lf = np.log(echelles[ok].astype(float)), np.log(F[ok])
    h, b = pente_mco(ls, lf)
    centres, pentes = pentes_glissantes(ls, lf, fenetre)
    moitie = len(ls) // 2
    h_bas = pente_mco(ls[:moitie], lf[:moitie])[0] if moitie >= 2 else np.nan
    h_haut = pente_mco(ls[-moitie:], lf[-moitie:])[0] if moitie >= 2 else np.nan
    return ResultatDFA(n, echelles[ok], F[ok], h, b, np.exp(centres), pentes, h_bas, h_haut)


def dfa_h(serie, echelle_min=DFA_ECHELLE_MIN, diviseur_max=DFA_DIVISEUR_ECHELLE_MAX,
          nb_echelles=N_S, ordre=DFA_ORDRE):
    """Pente globale seule (chemin rapide) ; nan si non calculable."""
    x = np.asarray(serie, dtype=float)
    echelles = echelles_dfa(len(x), echelle_min, diviseur_max, nb_echelles)
    if len(echelles) < 2:
        return float("nan")
    lf = log_fluctuations(fonction_fluctuation(x, echelles, ordre))
    return float(pentes_par_ligne(np.log(echelles.astype(float)), lf)[0])
