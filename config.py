# ============================================================
# config.py — Paramètres globaux du pipeline
# ------------------------------------------------------------
# V5 : chaque constante "X" a maintenant (quand c'est pertinent)
# une variante "X_MIN" (plancher absolu) et/ou un "X_RELATIF" /
# "X_FRACTION" qui exprime son seuil EFFECTIF = f(contexte du
# segment). La valeur absolue est CONSERVEE comme garde-fou.
#
# Les noms V4 sont conserves en ALIAS (deprecated) pour ne pas
# casser journal.py, courbe.py, filtres.py pendant la migration.
# Chaque alias porte un commentaire "# DEPRECATED -> voir X_MIN".
#
# Toutes les justifications (reference litteraire, choix
# empirique) sont dans documentation/CONSTANTES.md.
# ============================================================

# Seuil temporel pour la segmentation (en secondes)
# Si gap > TIME_GAP_THRESHOLD → nouveau segment
# NOTE : 600 s est un choix CONSERVATEUR pour preserver la
# continuite. La litterature GeoLife (guide officiel) parle
# d'un pas median ~7 s ; un gap de 10 min est donc une
# interruption majeure. Pas de reference standard pour ce seuil.
TIME_GAP_THRESHOLD = 900        # 10 minutes

# Nombre minimum de points pour qu'un segment soit conserve
# NOTE : 300 est un COMPROMIS rendement corpus GeoLife. La
# litterature DFA (Eke 2002, Gulich & Zunino 2014) recommande
# >= 2500 points pour une estimation fiable de H. On assume
# explicitement ce choix (voir 02_validation_methodes.py).
MIN_POINTS = 300

# Nombre minimum de points pour qu'un MORCEAU (issu d'une coupure
# aux artefacts, cf. segmentation.couper_aux_points_marques) soit
# exploitable par DFA.
# BUG V4 CORRIGE : h_dfa_par_morceaux utilisait N_MIN_WHITTLE (=20)
# comme seuil, ce qui permettait a DFA de "reussir" sur 2-3 echelles
# seulement -> H tres bruite. 50 est le minimum observe pour une
# regression DFA stable sur >= 5 echelles.
MIN_POINTS_MORCEAU = 50

# Vitesse maximale acceptable entre deux points GPS (en km/h)
# Garde-fou ABSOLU de dernier recours (T-Drive : v_max=27000 km/h).
# Le seuil EFFECTIF est max(MAX_SPEED, MAX_SPEED_RELATIF * v_med).
# Fourchette litterature : 161-300 km/h selon les etudes.
MAX_SPEED = 200.0               # km/h (plancher absolu)
MAX_SPEED_RELATIF = 20.0        # NOUVEAU : k tel que seuil = k * v_med
                                # (un segment a 3 km/h ne devrait pas
                                # tolerer un point a 60 km/h)

# Distance minimale parcourue pour qu'un segment soit conserve (en km)
# Si DISTANCE_MIN_RELATIF = True, le seuil effectif est calcule
# PAR FICHIER a partir du pas median observe :
#   seuil = max(DISTANCE_MIN,
#               MIN_POINTS * pas_median_s * V_PLANCHER_KMH / 3600)
# Un fichier a 5 s/pas et un fichier a 60 s/pas n'ont pas le meme
# sens physique pour le meme "0.1 km".
DISTANCE_MIN = 0.1              # km (plancher absolu)
DISTANCE_MIN_RELATIF = True     # NOUVEAU
DISTANCE_MIN_V_PLANCHER_KMH = 1.0   # NOUVEAU : v de "marche lente"

# Nombre de valeurs de s à générer pour la méthode DFA
N_S = 20

# Parametre pour la methode R/S
# m_min effectif = max(M_MIN_RS, N // M_MIN_RS_DIVISEUR)
# La litterature (these de simulation N=1000-4000) donne un
# optimal ~N/50. On prend N/20 (plus conservateur, moins de
# petits blocs bruites).
M_MIN_RS = 10                   # plancher absolu
M_MIN_RS_DIVISEUR = 20          # NOUVEAU : m_min = max(10, N // 20)

# Nombre de valeurs de m a generer pour la methode R/S
N_M = 20

# Nombre de points sur la grille de minimisation pour Whittle
K_WHITTLE = 1000

# Nombre minimum de points pour appliquer Whittle
# n_min effectif = max(N_MIN_WHITTLE,
#                      ceil(N_MIN_WHITTLE_M_CIBLE ** (1/0.65)))
# Avec m_cible=8 : n_min = 25 (au lieu de 20). m=8 garantit un
# ecart-type asymptotique de H ~ 1/sqrt(4*8) ~ 0.18 (raisonnable).
N_MIN_WHITTLE = 20              # plancher technique absolu
N_MIN_WHITTLE_M_CIBLE = 8       # NOUVEAU


# ============================================================
# Phase 3 du pipeline revise : filtres intra-segment (marquage)
# ============================================================

# --- Filtre distance (6a) : seuil RELATIF a la mediane du segment ---
# La litterature sur les outliers de trajectoire utilise des seuils
# relatifs (multiplicateur de la mediane) plutot qu'absolus : un
# saut de 400 m dans un segment de marche est plus aberrant qu'un
# saut de 400 m dans un segment de TGV.
# En mode "relatif" : seuil_effectif = min(MAX_DIST_M * mediane,
#                                          MAX_DIST_M_ABSOLU)
# (le min() garantit qu'on ne laisse JAMAIS passer au-dela du
#  plafond absolu, meme sur un segment a tres grand pas).
MODE_MAX_DIST = "relatif"       # "absolu" | "relatif"  (V4="absolu")
MAX_DIST_M = 30                 # en mode "relatif" : 30 x distance_mediane (m)
MAX_DIST_M_ABSOLU = 500         # NOUVEAU : plafond absolu (m)

# Ordre d'application des filtres 6a (distance) et 6b (vitesse)
ORDRE_FILTRES = "distance_puis_vitesse"


# ============================================================
# Filtre MAD (remplace filtrer_vitesse absolu) — Phase 3 revisee
# ============================================================

# Z-score modifie asymetrique — INCHANGE (soutenu par la litterature).
# Reference : Iglewicz & Hoaglin (1993), "Volume 16: How to Detect
# and Handle Outliers", ASQC Quality Press. Le facteur 0.6745
# (= Phi^-1(3/4)) est mathematique, PAS ajustable.
MAD_Z_SEUIL = 3.5

# Persistance minimale d'un cluster de candidats pour parler de
# "regime" (changement de mode) plutot que de "bruit local".
# Seuil effectif = max(MAD_PERSISTANCE_MIN,
#                      round(MAD_PERSISTANCE_FRACTION * N))
# Pour N=300 : max(5, 3) = 5. Pour N=3000 : max(5, 30) = 30.
MAD_PERSISTANCE_MIN = 5         # plancher absolu (points)
MAD_PERSISTANCE_FRACTION = 0.01 # NOUVEAU : 1% de la longueur

# Variation locale de vitesse au-dela de laquelle un pic est
# classe "brutal" (artefact) plutot que "progressif" (regime).
# Seuil effectif = max(MAD_DELTA_V_BRUTAL,
#                      MAD_DELTA_V_FACTEUR_MAD * MAD_v)
# Un segment a MAD_v=1 km/h ne devrait pas utiliser le meme
# seuil qu'un segment a MAD_v=15 km/h.
MAD_DELTA_V_BRUTAL = 20.0       # km/h (plancher absolu)
MAD_DELTA_V_FACTEUR_MAD = 5.0   # NOUVEAU

# Alerte globale d'heterogeneite : ratio R = sigma_v / v_mediane.
# Deux modes possibles :
#   - MAD_R_TEST_NUL = False : comparaison au seuil absolu 1.5
#   - MAD_R_TEST_NUL = True  : comparaison au quantile 95% de la
#                              distribution nulle (via
#                              validation/modeles_nuls.py)
MAD_R_HETEROGENE = 1.5          # seuil absolu (fallback)
MAD_R_TEST_NUL = True           # NOUVEAU : tester contre la loi nulle

# Fallback quand MAD = 0 (segment quasi-immobile).
# Seuil effectif = max(MAD_FALLBACK_SPEED,
#                      MAD_FALLBACK_SPEED_FACTEUR * v_mediane)
MAD_FALLBACK_SPEED = 5.0        # km/h (plancher absolu)
MAD_FALLBACK_SPEED_FACTEUR = 3.0    # NOUVEAU


# ============================================================
# Stationnarite de la serie des distances (Phase 2 etendue)
# ============================================================
# V5 : remplace le ratio heuristique (premiere moitie / deuxieme
# moitie) par un test KS 2-echantillons. Le ratio est CONSERVE
# pour l'AFFICHAGE dans le log (indicateur rapide a l'oeil), mais
# il ne pilote plus la decision.
STATIO_TEST = "ks"              # NOUVEAU : "ks" | "ratio" | "none"
STATIO_ALPHA = 0.05             # NOUVEAU : seuil du test KS
STATIO_MIN_POINTS = 10          # NOUVEAU : min points par moitie
                                # (remplace le hardcode len>=4)
# Ratio conserve pour l'affichage uniquement (non prescriptif)
STATIO_RATIO_MIN = 0.7
STATIO_RATIO_MAX = 1.4


# ============================================================
# Journalisation (logs texte par segment) — pretraitement/journal.py
# ============================================================
# Un gap est signale comme "notable" si son delta_t depasse
#   max(SEUIL_GAP_NOTABLE_MIN,
#       SEUIL_GAP_NOTABLE_RELATIF * pas_median)
# V4 utilisait 5 s ABSOLU, ce qui est INFERIEUR au pas median
# GeoLife (~7 s) -> la moitie des intervalles normaux etaient
# signales "gap". Le seuil relatif corrige ce probleme.
SEUIL_GAP_NOTABLE_MIN = 5       # secondes (plancher absolu)
SEUIL_GAP_NOTABLE_RELATIF = 2.5 # NOUVEAU : x pas_median

# ALIAS DEPRECATED : conserve pour ne pas casser journal.py et
# courbe.py pendant la migration. A supprimer une fois que les
# deux modules utilisent SEUIL_GAP_NOTABLE_MIN/RELATIF.
SEUIL_GAP_NOTABLE = SEUIL_GAP_NOTABLE_MIN   # DEPRECATED

# Seuils d'observation qualitative dans le tableau des phases
# (utilises UNIQUEMENT pour le champ "observations" du log,
# pas pour aucune decision algorithmique).
JOURNAL_SEUILS_VITESSE_KMH = (1.0, 8.0, 20.0, 50.0)     # NOUVEAU
JOURNAL_LABELS_VITESSE = (
    "arret / quasi-immobile",
    "marche probable",
    "velo / marche rapide probable",
    "vehicule lent (bus, voiture urbaine)",
    "vehicule rapide / suspect",
)


# ============================================================
# AJOUTS V5 -- estimation rapide, modele nul, echantillonnage
# ============================================================

# Graine unique pour toute simulation / test nul (reproductibilite)
GRAINE = 20261018

# Serie analysee par les estimateurs : "distance" (d_i, en metres) ou "vitesse"
SERIE_ANALYSEE = "distance"

# --- DFA rapide ---
DFA_ECHELLE_MIN = 4             # V4 (Peng 1994). A tester {4,6,8,10}
DFA_DIVISEUR_ECHELLE_MAX = 4    # s_max = N // 4
DFA_ORDRE = 1                   # DFA-1 (standard)
DFA_FENETRE_PENTES_LOCALES = 5
DFA_N_MIN = 16                  # NOUVEAU : remplace 2 hardcodes dans dfa.py

# --- R/S rapide ---
RS_DIVISEUR_M_MAX = 2           # m_max = N // 2

# --- Whittle local rapide ---
WHITTLE_EXPOSANT_M = 0.65       # m = min(N ** 0.65, N // 4)
WHITTLE_DIVISEUR_M_MAX = 4

# --- Modele nul a memoire courte (validation/modeles_nuls.py) ---
AVEC_TEST_NUL = True
NUL_MODELE = "ar"
NUL_NB_SURROGATS = 99
NUL_ORDRE_AR_MAX = 5
NUL_ALPHA = 0.05

# --- Part "immobile" dans estimation_segment.py ---
# V4 : d < 1.0 m ABSOLU. V5 : seuil relatif a la mediane si None.
PART_IMMOBILE_SEUIL_M = None    # NOUVEAU : None -> 0.1 * mediane(d)

# --- Mode d'enregistrement (echantillonnage.py) ---
ECHANT_TOL_DT = 0.2
ECHANT_TOL_D = 0.3
ECHANT_PART_TEMPS = 0.6
ECHANT_PART_DISTANCE = 0.5
ECHANT_DEPLACEMENT_MIN_M = 1.0  # plancher absolu pour "en mouvement"
ECHANT_MIN_MOUVEMENT = 10       # NOUVEAU : remplace hardcode >= 10

# --- Courbes ---
COURBE_DEUX_PANNEAUX = False

# ============================================================
# Detection d'arrets et variables de la loi de puissance
# (pretraitement/arrets.py, experiences/lois_de_puissance.py)
#
# ATTENTION : les seuils d'arret ne sont PAS universels
# (voir documentation/LITTERATURE.md). Deux ecoles existent
# dans la litterature GPS :
#   - 100 m / 5 min  (defaut trackintel, tres courant)
#   - 200 m / 20-30 min (Zheng et al., tres utilise sur
#     Geolife, qui est le corpus de ce memoire)
# Le choix par defaut ci-dessous suit Zheng et al. pour
# Geolife ; il est faire varier via ARRET_JEUX_SENSIBILITE
# pour mesurer la sensibilite du verdict.
# ============================================================

# Rayon (m) et duree minimale (s) d'un arret : fenetre
# glissante de Li et al. (2008).
# 200 m / 20 min = choix Zheng et al., adapte a Geolife.
ARRET_RAYON_M = 200.0
ARRET_DUREE_MIN_S = 1200.0

# Un intervalle entre deux points GPS plus long que ceci
# coupe la serie en morceaux independants (on ne raccorde
# pas). 15 min = choix de trackintel (gap_threshold).
ARRET_TROU_MAX_S = 900.0

# Une pause entre deux segments consecutifs n'est retenue
# que si la fin du premier segment et le debut du second
# sont a moins de ce rayon (m).
# CHOIX DE ROBUSTESSE : aligne sur ARRET_RAYON_M pour ne
# pas couper des pauses coherentes. A documenter dans le
# memoire comme choix arbitraire.
ARRET_RAYON_PAUSE_M = 200.0

# Distance minimale (m) entre deux arrets consecutifs pour
# qu'elle soit consideree comme un deplacement.
# CHOIX DE ROBUSTESSE : 100 m < ARRET_RAYON_M, donc deux
# arrets separes de moins de 100 m sont "au meme endroit".
# A documenter comme choix arbitraire.
ARRET_DISTANCE_MIN_M = 100.0

# Jeux de seuils (rayon_m, duree_min_s) essayes pour
# mesurer la sensibilite du verdict au choix des seuils.
# Liste explicitement donnee dans le docstring d'arrets.py :
#   (100 m, 5 min), (100 m, 10 min),
#   (200 m, 20 min), (200 m, 30 min)
ARRET_JEUX_SENSIBILITE = (
    (100.0, 300.0),
    (100.0, 600.0),
    (200.0, 1200.0),
    (200.0, 1800.0),
)

# Nombre minimal d'observations (toutes variables confondues,
# pas seulement la queue) pour analyser un utilisateur isole
# dans experiences/lois_de_puissance.py.
# CHOIX DE ROBUSTESSE : doit etre >= LP_NB_MIN_QUEUE pour que
# la queue minimale puisse exister. 100 = 2 x LP_NB_MIN_QUEUE,
# cohérent avec le fait que la queue fait typiquement <= 50 %
# des observations. A verifier empiriquement : si beaucoup
# d'utilisateurs sont sous ce seuil sur Geolife, le baisser
# et le signaler comme limite.
LP_NB_MIN_PAR_UTILISATEUR = 100

# ============================================================
# Lois de puissance — méthode de Clauset et al. (2009)
# Adapté au contexte mobilité humaine (Geolife, NYC Taxi)
#
# ATTENTION : les quatre constantes ci-dessous ne sont PAS
# toutes des prescriptions de Clauset et al. Voir le
# commentaire de chacune pour savoir ce qui est sourcé et
# ce qui est un choix de robustesse à justifier dans le
# mémoire.
# ============================================================

# Seuil de p-valeur pour l'adequation (verdict "rejetee").
# SOURCE : Clauset, Shalizi & Newman (2009, §5) : p <= 0.1
#   => loi de puissance rejetee ; p > 0.1 => non rejetee
#   (plausible, PAS prouvee). Convention reprise dans la
#   quasi-totalite des applications.
LP_SEUIL_P = 0.1

# Nombre de simulations bootstrap pour la p-valeur d'adequation.
# CHOIX DE ROBUSTESSE (pas une prescription de Clauset) :
#   l'erreur type de p_hat est bornee par 0.5 / sqrt(B).
#   B = 1000 => erreur type <= 0.016 ; B = 2500 => <= 0.01.
#   A verifier empiriquement (500 / 1000 / 2000 sur une ou
#   deux distributions representatives) : si le verdict au
#   seuil LP_SEUIL_P est stable, 1000 suffit.
# NOTE : le package powerlaw (Alstott et al. 2014)
#   N'IMPLEMENTE PAS ce bootstrap (issue #29 du depot) :
#   le nombre de simulations est donc un choix explicite
#   que le memoire doit documenter.
LP_NB_BOOTSTRAP = 1000

# Taille minimale de la queue (n_queue = nombre de points
# avec x >= xmin) pour qu'un ajustement soit juge fiable.
# CHOIX DE ROBUSTESSE, PAS un standard documente :
#   une queue de quelques dizaines de points rend alpha
#   instable et la selection de xmin (par KS) peu fiable.
#   50 est un garde-fou conservateur, a verifier par
#   analyse de sensibilite (50 / 100 / 200) : si alpha
#   varie de plus de 0.1 entre deux valeurs, la queue est
#   trop courte pour une conclusion ferme (marquer la
#   distribution "queue_insuffisante" plutot que de forcer
#   un ajustement).
LP_NB_MIN_QUEUE = 50

# Nombre maximal de valeurs candidates de xmin essayees.
# COMPROMIS COUT / PRECISION (pas une prescription) :
#   le package powerlaw utilise TOUTES les valeurs uniques
#   de l'echantillon comme candidats (find_xmin). C'est
#   exact mais couteux : le bootstrap refait la recherche
#   des centaines de fois. 200 candidats equi-repartis
#   changent peu le xmin retenu si la courbe KS(xmin) est
#   lisse autour du minimum.
# A verifier empiriquement sur une ou deux distributions
#   representatives : comparer xmin et alpha obtenus avec
#   200 candidats vs toutes les valeurs uniques ; si
#   l'ecart relatif sur alpha reste < 2%, 200 est suffisant.
LP_NB_CANDIDATS_XMIN = 200


# ============================================================
# Tracabilite : empreinte de tous les reglages
# ============================================================
def empreinte_config():
    """
    Empreinte courte (10 caracteres) de TOUTES les constantes de ce fichier.
    Ecrite dans les resultats : on sait toujours avec quels reglages un chiffre a ete produit.
    Changer une seule constante change l'empreinte.

    NOTE V5 : les alias DEPRECATED (SEUIL_GAP_NOTABLE) sont aussi
    inclus dans l'empreinte. C'est volontaire : changer un alias
    change l'empreinte, ce qui signale qu'on a modifie config.py
    meme si le comportement effectif est identique.
    """
    import hashlib
    import json
    valeurs = {nom: valeur for nom, valeur in globals().items()
               if nom.isupper() and not nom.startswith("_")}
    texte = json.dumps(valeurs, sort_keys=True, default=str)
    return hashlib.sha1(texte.encode("utf-8")).hexdigest()[:10]