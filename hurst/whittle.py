# ./hurst/whittle.py
#
# Methode de Whittle pour l'estimation du parametre de Hurst
#
# Fusionne whittle/spectral.py et whittle/hurst.py en un seul fichier.
# Le module commun (extraction_distances) vient de hurst/serie_distances.py.
#
# Reference : Whittle, P. (1953). Estimation and information in stationary
#             time series. Arkiv for Matematik, 2(5), 423-434.

from math  import pi, floor, log, fabs, sqrt
import numpy as np

from hurst.serie_distances import extraction_distances
from config import K_WHITTLE, N_MIN_WHITTLE


# ---------------------------------------------------------------
# Calcul du periodogramme
# ---------------------------------------------------------------

def frequences(N):
    """
    Calcule les frequences de Fourier discretes

    Les frequences sont definies pour que les sinusoides soient
    orthogonales sur [1, N]. La frequence j correspond a exactement
    j cycles complets sur la serie de longueur N.

    Formule : lambda_j = 2*pi*j / N, j = 1, ..., floor(N/2)

    Entree :
        N : longueur de la serie

    Sortie :
        liste de float : frequences lambda_j en radians
    """
    liste_lambdas = []

    for j in range(1, N // 2 + 1):
        liste_lambdas.append((2 * pi * j) / N)

    return liste_lambdas


def periodogramme(X, lambdas):
    """
    Calcule le periodogramme de la serie X aux frequences lambdas

    Le periodogramme est une estimation de la densite spectrale de
    puissance. Pour chaque frequence lambda_j :

        Aj = cte * sum_{k=1}^{N} x_k * cos(k * lambda_j)
        Bj = cte * sum_{k=1}^{N} x_k * sin(k * lambda_j)
        I(lambda_j) = Aj^2 + Bj^2

    avec cte = 1 / sqrt(2 * pi * N)

    OPTIMISATION : la boucle interne sur k est vectorisee avec numpy
    (produit scalaire) pour eviter O(N^2) operations avec deux boucles
    Python imbriquees. La formule mathematique est identique.

    Entree :
        X       : serie des distances (liste de float)
        lambdas : liste des frequences de Fourier

    Sortie :
        I       : liste de float, valeurs du periodogramme
        lambdas : liste de float, frequences correspondantes
    """
    N   = len(X)
    cte = 1.0 / sqrt(2 * pi * N)

    # Conversion en tableau numpy pour la vectorisation
    X_arr = np.array(X)
    k_arr = np.arange(1, N + 1)   # k = 1, ..., N

    I = []

    for lam in lambdas:
        # Produit scalaire vectorise : equivalent a la boucle sur k
        Aj = cte * np.dot(X_arr, np.cos(k_arr * lam))
        Bj = cte * np.dot(X_arr, np.sin(k_arr * lam))
        I.append(float(Aj * Aj + Bj * Bj))

    return I, lambdas


# ---------------------------------------------------------------
# Critere de Whittle
# ---------------------------------------------------------------

def nb_frequences(N):
    """
    Calcule le nombre de basses frequences utilisees dans le critere

    On utilise uniquement les m premieres frequences de Fourier car
    la loi de puissance f(lambda) ~ |lambda|^(1-2H) n'est valide
    qu'aux basses frequences (lambda -> 0).

    Formule standard : m = floor(N^0.65)
    Limite a N//4 pour rester dans le regime basse frequence
    meme sur les petits segments.

    Entree :
        N : longueur de la serie

    Sortie :
        int : nombre de frequences m
    """
    return min(floor(pow(N, 0.65)), N // 4)


def critere_whittle(H, lambdas, I, m):
    """
    Evalue le critere de Whittle reduit pour une valeur de H donnee

    Apres elimination analytique de la constante C, le critere devient :

        L(H) = log(C_hat(H)) + (1 - 2H) * (1/m) * sum_j log|lambda_j|

    avec :
        C_hat(H) = (1/m) * sum_{j=1}^{m} I(lambda_j) / |lambda_j|^(1-2H)

    Plus L(H) est petit, meilleur est l'ajustement.

    Entree :
        H       : float, valeur candidate de H dans (0, 1)
        lambdas : liste des frequences de Fourier
        I       : liste des valeurs du periodogramme
        m       : nombre de basses frequences a utiliser

    Sortie :
        float : valeur du critere L(H)
    """
    C_hat = 0
    somme = 0

    for j in range(m):
        C_hat += I[j] / pow(fabs(lambdas[j]), 1 - 2 * H)
        somme += log(fabs(lambdas[j]))

    return log(C_hat / m) + (1 - 2 * H) * somme / m


# ---------------------------------------------------------------
# Point d'entree principal
# ---------------------------------------------------------------

def Whittle(segment):
    """
    Applique la methode de Whittle sur un segment et retourne H

    Accepte deux formats (detection automatique) :
        - Liste de points GPS [[lat, lon, ...], ...]
        - Serie de floats [d1, d2, ...] (fBm synthetique)

    Pipeline :
        1. Extraire les distances (ou utiliser la serie directe)
        2. Calculer les frequences de Fourier
        3. Calculer le periodogramme (vectorise)
        4. Determiner le nombre de basses frequences m
        5. Minimiser le critere de Whittle sur une grille de H

    Minimisation par recherche sur grille :
        H_k = k / (K+1), k = 1, ..., K  avec K = K_WHITTLE = 1000
    On retient le H_k qui minimise le critere.

    NOTE sur le blocage a 0.999 : si le minimum se trouve en bout
    de grille (H_k = K/(K+1) ≈ 0.999), cela indique que la vraie
    solution serait H >= 1, signe d'une serie contaminee par un
    saut GPS. Ce n'est pas un bug mais un signal diagnostique.

    Entree :
        segment : liste de points GPS ou liste de float

    Sortie :
        float : estimation de H dans (0, 1)
        None  : si le segment est trop court
    """
    distances = extraction_distances(segment)
    N         = len(distances)

    if N < N_MIN_WHITTLE:
        print(f"[AVERTISSEMENT] Segment trop court pour Whittle : N={N}")
        return None

    liste_lambdas     = frequences(N)
    I, lambdas        = periodogramme(X=distances, lambdas=liste_lambdas)

    m = nb_frequences(N)

    if m < 2:
        print(f"[AVERTISSEMENT] Pas assez de frequences pour Whittle : m={m}")
        return None

    # Recherche du H qui minimise le critere sur la grille
    H_optimal = 1 / (K_WHITTLE + 1)
    L_min     = critere_whittle(H_optimal, lambdas, I, m)

    k = 2
    while k <= K_WHITTLE:
        H_k = k / (K_WHITTLE + 1)
        L_k = critere_whittle(H_k, lambdas, I, m)

        if L_k < L_min:
            L_min     = L_k
            H_optimal = H_k

        k += 1

    return H_optimal


# ============================================================================
# VERSION RAPIDE (V5)
# ----------------------------------------------------------------------------
# Meme calcul que Whittle() ci-dessus (REFERENCE LISIBLE), vectorise avec numpy
# (periodogramme par FFT, critere evalue sur toute la grille d'un coup).
#
# IMPORTANT -- ce que fait reellement cet estimateur : c'est le Whittle LOCAL de
# Robinson (1995). Pres de la frequence nulle on suppose f(lambda) ~ C * lambda^(1-2H)
# et on minimise le critere sur les m premieres frequences de Fourier seulement.
# Ce n'est PAS un Whittle sur [-pi, pi] avec la densite complete du fGn.
# Variance asymptotique de H : 1/(4m). Avec m = N^0.65 et N = 100, m = 19, donc
# ecart-type ~ 0.115 : c'est une limite de la methode, pas un defaut du code.
#
# a_la_borne = True quand le minimum est a une borne de la grille (H ~ 0 ou H ~ 1) :
# l'estimation n'est alors pas fiable (serie probablement non stationnaire).
# ============================================================================
from dataclasses import dataclass

import numpy as np

from config import WHITTLE_EXPOSANT_M, WHITTLE_DIVISEUR_M_MAX


def periodogramme_rapide(x):
    """Frequences de Fourier lambda_j = 2*pi*j/N et periodogramme I_j, pour j = 1..N//2 (par FFT)."""
    n = len(x)
    f = np.fft.rfft(np.asarray(x, dtype=float))
    j = np.arange(1, n // 2 + 1)
    return 2 * pi * j / n, (np.abs(f[1:n // 2 + 1]) ** 2) / (2 * pi * n)


@dataclass(frozen=True)
class ResultatWhittle:
    n: int
    m: int
    h: float
    a_la_borne: bool


def whittle_local(serie, grille=K_WHITTLE, n_min=N_MIN_WHITTLE,
                  exposant_m=WHITTLE_EXPOSANT_M, diviseur_m=WHITTLE_DIVISEUR_M_MAX):
    """Whittle local rapide : H sur la grille k/(grille+1), k = 1..grille ; nan si non calculable."""
    x = np.asarray(serie, dtype=float)
    n = len(x)
    if n < n_min:
        return ResultatWhittle(n, 0, np.nan, False)
    m = min(floor(n ** exposant_m), n // diviseur_m)
    if m < 2:
        return ResultatWhittle(n, m, np.nan, False)
    lam, I = periodogramme_rapide(x)
    lam, I = lam[:m], I[:m]
    log_lam = np.log(lam)
    H = np.arange(1, grille + 1) / (grille + 1)
    expo = (2 * H - 1)[:, None] * log_lam[None, :]                          # (grille, m)
    critere = np.log((np.exp(expo) * I[None, :]).mean(axis=1)) - (2 * H - 1) * log_lam.mean()
    k = int(np.argmin(critere))                                             # premier minimum en cas d'egalite
    return ResultatWhittle(n, m, float(H[k]), k in (0, grille - 1))
