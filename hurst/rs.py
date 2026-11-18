# ./hurst/rs.py
#
# Methode R/S (Rescaled Range) pour l'estimation du parametre de Hurst
#
# Fusionne rs/rs.py et rs/hurst.py en un seul fichier.
# Le module commun (extraction_distances) vient de hurst/serie_distances.py.
#
# Reference : Hurst, H.E. (1951). Long-term storage capacity of reservoirs.
#             Transactions of the American Society of Civil Engineers, 116, 770-799.

from math import log, exp, floor, sqrt

from hurst.serie_distances import extraction_distances
from config  import N_M, M_MIN_RS


# ---------------------------------------------------------------
# Generation des tailles de blocs m
# ---------------------------------------------------------------

def generer_valeurs_m(N, n_m):
    """
    Genere les tailles de blocs m pour la methode R/S

    Les valeurs sont espacees exponentiellement entre m_min et m_max
    pour obtenir des points regulierement repartis en echelle log-log.

    Formule pour le j-ieme point (j = 1, ..., n_m) :
        log(m_j) = log(m_min) + ((j-1) / (n_m-1)) * (log(m_max) - log(m_min))
        m_j = floor(exp(log(m_j)))

    Contraintes :
        m_min = M_MIN_RS = 10  (ecart-type trop bruite en dessous)
        m_max = N // 2         (garantit au moins 2 blocs complets)

    Entree :
        N   : longueur de la serie (nombre de distances)
        n_m : nombre de valeurs a generer avant suppression des doublons

    Sortie :
        liste d'entiers : valeurs distinctes de m en ordre croissant
        liste vide si N trop court
    """
    m_min = M_MIN_RS
    m_max = N // 2

    # Garde : si m_max <= m_min, la serie est trop courte
    if m_max <= m_min:
        return []

    ensemble_m = []

    for j in range(1, n_m + 1):
        log_mj = log(m_min) + ((j - 1) / (n_m - 1)) * (log(m_max) - log(m_min))
        mj     = floor(exp(log_mj))
        ensemble_m.append(mj)

    # Suppression des doublons (valeurs identiques apres floor)
    ensemble_m_unique = []
    for m in ensemble_m:
        if m not in ensemble_m_unique:
            ensemble_m_unique.append(m)

    return ensemble_m_unique


# ---------------------------------------------------------------
# Calcul de la statistique R/S
# ---------------------------------------------------------------

def rs_bloc(m, X, indice_debut):
    """
    Calcule la statistique R/S sur un bloc de taille m

    Soit le bloc X[indice_debut .. indice_debut+m-1] :
        1. Moyenne locale moy
        2. Ecarts cumules W(k) = sum_{i=0}^{k-1} (x_i - moy)
        3. Plage R = max(W) - min(W)
        4. Ecart-type S = sqrt(variance locale biaisee, diviseur m)
        5. Statistique R/S = R / S

    Note sur le diviseur : on utilise m (variance biaisee) conformement
    a la definition classique de Hurst (1951).

    Entree :
        m           : taille du bloc
        X           : serie complete des distances
        indice_debut: indice de debut du bloc dans X

    Sortie :
        float : statistique R/S du bloc
        None  : si le bloc est stationnaire (S = 0)
    """
    # Moyenne locale
    moyenne_locale = 0
    for i in range(indice_debut, indice_debut + m):
        moyenne_locale += X[i]
    moyenne_locale = moyenne_locale / m

    # Ecarts cumules et somme des carres pour l'ecart-type
    ecarts_cumules = []
    cumul          = 0
    somme_carres   = 0

    for i in range(m):
        ecart         = X[indice_debut + i] - moyenne_locale
        cumul        += ecart
        somme_carres += ecart * ecart
        ecarts_cumules.append(cumul)

    plage      = max(ecarts_cumules) - min(ecarts_cumules)
    ecart_type = sqrt(somme_carres / m)

    if ecart_type == 0:
        return None   # bloc stationnaire, R/S indefini

    return plage / ecart_type


def moyenne_rs(m, X):
    """
    Calcule la moyenne de R/S sur tous les blocs de taille m

    Decoupe X en blocs non chevauchants de taille m.
    Calcule rs_bloc() sur chaque bloc et moyenne les resultats.
    Les blocs stationnaires (R/S = None) sont ignores.

    CORRECTION : on divise par le nombre de blocs VALIDES
    (et non par le nombre total de blocs, ce qui sous-estimerait
    la moyenne si des blocs sont stationnaires).

    Entree :
        m : taille du bloc
        X : serie complete des distances

    Sortie :
        float : moyenne de R/S pour la taille m
        None  : si aucun bloc n'est valide
    """
    taille     = len(X)
    indice     = 0
    somme      = 0
    nb_valides = 0   # compteur de blocs effectivement utilises

    while (indice + m) <= taille:
        rs = rs_bloc(m, X, indice)
        if rs is not None:
            somme      += rs
            nb_valides += 1
        indice += m

    if nb_valides == 0:
        return None

    return somme / nb_valides


def paires_m_rs(ensemble_m, X):
    """
    Calcule les paires (m, moyenne_RS(m)) pour chaque valeur de m

    Filtre les paires invalides (moyenne_RS = None ou = 0)
    pour eviter log(0) dans la regression.

    Entree :
        ensemble_m : liste des tailles de blocs
        X          : serie complete des distances

    Sortie :
        liste de tuples (m, moyenne_RS(m))
    """
    paires = []

    for m in ensemble_m:
        rs_moy = moyenne_rs(m, X)

        # log(0) interdit et log(None) plante -> on filtre
        if rs_moy is not None and rs_moy > 0:
            paires.append((m, rs_moy))

    return paires


# ---------------------------------------------------------------
# Regression log-log et estimation de H
# ---------------------------------------------------------------

def estimer_H_regression(paires):
    """
    Estime le parametre de Hurst H par regression lineaire log-log

    Soit U_j = log(m_j) et V_j = log(RS_moyen(m_j)).
    On cherche la pente H de la droite V = H*U + C par moindres carres :

        H = (n * sum(Uj*Vj) - sum(Uj) * sum(Vj)) /
            (n * sum(Uj^2)  - sum(Uj)^2)

    Entree :
        paires : liste de tuples (m, RS_moyen(m))

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

def RS(segment):
    """
    Applique la methode R/S complete sur un segment et retourne H

    Accepte deux formats (detection automatique) :
        - Liste de points GPS [[lat, lon, ...], ...]
        - Serie de floats [d1, d2, ...] (fBm synthetique)

    Pipeline :
        1. Extraire les distances entre points successifs
        2. Generer les tailles de blocs m (log-espaces)
        3. Calculer les paires (m, moyenne_RS(m))
        4. Estimer H par regression log-log

    Entree :
        segment : liste de points GPS ou liste de float

    Sortie :
        float : estimation de H
        None  : si le segment est trop court ou l'estimation echoue
    """
    distances = extraction_distances(segment)
    N         = len(distances)

    if N < 2 * M_MIN_RS:
        print(f"[AVERTISSEMENT] Segment trop court pour R/S : N={N}")
        return None

    ensemble_m = generer_valeurs_m(N, N_M)

    if len(ensemble_m) < 2:
        print(f"[AVERTISSEMENT] Pas assez de valeurs de m pour R/S : N={N}")
        return None

    paires = paires_m_rs(ensemble_m, distances)

    if len(paires) < 2:
        print(f"[AVERTISSEMENT] Pas assez de paires valides pour la regression R/S")
        return None

    H = estimer_H_regression(paires)

    return H


# ============================================================================
# VERSION RAPIDE (V5)
# ----------------------------------------------------------------------------
# Meme calcul que RS() ci-dessus (REFERENCE LISIBLE), vectorise avec numpy, plus
# une estimation CORRIGEE du biais de petit echantillon (Anis & Lloyd, 1976 ;
# Weron, 2002) : sur du bruit sans memoire, le R/S brut donne ~0.55 au lieu de 0.50.
#   h_brut     : pente brute (celle de RS() ci-dessus)
#   h_corrige  : 0.5 + pente de log(R/S) - log E[R/S sous bruit i.i.d.]
# La correction retire le biais a H = 0.5 mais sous-estime les series tres persistantes :
# la validation (02_validation_methodes.py) chiffre les deux, on rapporte les deux.
# tests/test_rs.py verifie la parite avec RS() ci-dessus.
# ============================================================================
from dataclasses import dataclass
from math import exp, lgamma, pi, sqrt

import numpy as np

from config import RS_DIVISEUR_M_MAX
from hurst.echelles import grille_echelles, pente_mco


def esperance_rs(n):
    """
    Esperance de R/S pour n observations i.i.d. gaussiennes (formule exacte d'Anis et Lloyd, 1976).

    Valable pour la definition utilisee ici : ecart-type avec diviseur n, amplitude du profil cumule
    des ecarts a la moyenne du bloc. Le facteur (n - 1/2)/n de Peters (1994) a ete teste et ecarte :
    il sous-estime l'esperance de 5 % a n = 10 pour cette definition (verifie par Monte-Carlo,
    cf. tests/test_rs.py::test_esperance_rs_correspond_a_monte_carlo).
    """
    i = np.arange(1, n)
    total = np.sqrt((n - i) / i).sum()
    if n <= 340:
        g = exp(lgamma((n - 1) / 2) - lgamma(n / 2)) / sqrt(pi)
    else:
        g = 1.0 / sqrt(n * pi / 2)      # approximation asymptotique de Gamma((n-1)/2) / Gamma(n/2)
    return g * total


def rs_moyen(x, m):
    """Moyenne de R/S sur les blocs disjoints de taille m (les blocs d'ecart-type nul sont ignores)."""
    nb_blocs = len(x) // m
    if nb_blocs < 1:
        return float("nan")
    B = x[:nb_blocs * m].reshape(nb_blocs, m)
    ecarts = B - B.mean(axis=1, keepdims=True)
    cumul = np.cumsum(ecarts, axis=1)
    amplitude = cumul.max(axis=1) - cumul.min(axis=1)
    sigma = np.sqrt((ecarts * ecarts).mean(axis=1))
    ok = sigma > 0
    if not ok.any():
        return float("nan")
    return float((amplitude[ok] / sigma[ok]).mean())


@dataclass(frozen=True)
class ResultatRS:
    n: int
    tailles: np.ndarray
    rs: np.ndarray
    h_brut: float
    h_corrige: float


def rs_complet(serie, m_min=M_MIN_RS, diviseur_max=RS_DIVISEUR_M_MAX, nb_echelles=N_M):
    """R/S complet : h_brut (comme la V4) et h_corrige (Anis-Lloyd). nan si non calculable."""
    x = np.asarray(serie, dtype=float)
    n = len(x)
    vide = np.array([])
    if n < 2 * m_min:
        return ResultatRS(n, vide.astype(int), vide, np.nan, np.nan)
    tailles = grille_echelles(m_min, n // diviseur_max, nb_echelles)
    if len(tailles) < 2:
        return ResultatRS(n, tailles, vide, np.nan, np.nan)
    valeurs = np.array([rs_moyen(x, int(m)) for m in tailles])
    ok = np.isfinite(valeurs) & (valeurs > 0)
    tailles, valeurs = tailles[ok], valeurs[ok]
    if len(tailles) < 2:
        return ResultatRS(n, tailles, valeurs, np.nan, np.nan)
    lm = np.log(tailles.astype(float))
    h_brut, _ = pente_mco(lm, np.log(valeurs))
    attendu = np.array([esperance_rs(int(m)) for m in tailles])
    pente_corrigee, _ = pente_mco(lm, np.log(valeurs) - np.log(attendu))
    return ResultatRS(n, tailles, valeurs, h_brut, 0.5 + pente_corrigee)
