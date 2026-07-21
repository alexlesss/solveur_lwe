import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sympy import Matrix
#from fpylll import IntegerMatrix, LLL

# Dans ce fichier il est possible de trouver plusieurs solveurs, tous avec de légères différences.

# Le premier solveur implémente directement la réduction que l'on présente en section 
# 5.1 du rapport. 
# Les paramètres d'entrées à la fonction solve_lwe sont les données accesibles
# publiquement depuis la construction d'une instance de LWE. On a donc un
# quadruplet (A,b,q,t) tel que:
# 
# A: C'est une matrice de taille m x n
# b: C'est un vecteur de taille m
# q: C'est le nombre premier qui régit notre système, comme nous travaillons
# toujours sous Z_q
# t: C'est la borne que l'on applique pour le vecteur d'erreur que l'on 
# cherche.
# Ces paramètres nous donnent accès à toute l'information nécessaire pour 
# résoudre le problème.
# Nous intialisons aussi un paramètre de limite de temps, qui sera fixé à 30 minutes, 
# donc 1800 secondes. Il est en paramètre pour une raison de flexibilité.
def solve_lwe(A,b,q,t, time_limit=1300):
    
    # Allons chercher la taille du problème pour commencer:
    m, n = A.shape
    # Comme nous utiliserons souvent la qte de taille m-n, posons la
    m_n = m - n 

    # Partitionnons maintenant A et b tel qu'en section 5.1.1
    A_0 = A[0:n, :]
    A_1 = A[n:m, :]
    b_0 = b[:n]
    b_1 = b[n:]

    # Inversons la matrice A_0:
    A_0inv = np.array(Matrix(A_0).inv_mod(q)).astype(int)

    # et depuis cette matrice inverse, calculons W et u.
    W = (A_1 @ A_0inv) % q
    u = (b_1 - W @ b_0) % q


    # Nous pouvons donc maintenant construire notre modèle! Rappelons que nos
    # faisons ici dire
    modele = gp.Model("SolveurLWE")
    modele.setParam('TimeLimit', time_limit)
    # Comme les problèmes sur lesquels nous allons travailler seront bien con-
    # ditionnés (nous présenterons ce conditionnement plus tard), nous pouvons
    # assumer qu'il n'y a qu'une seule solution valide.
    modele.setParam('SolutionLimit', 1)

    # Posons les variables de décision, soit les x_k qui sont bornés par t
    x = modele.addVars(m, vtype=GRB.INTEGER, lb=-t, ub=t, name="x")

    # Et les variables f alternatives pour le modulo, pour 
    # lesquelles ont doit calculer leurs bornes.
    f_inf = -1 * ((t*(n*q - n + 1) + q - 1) // q)
    f_sup = (t*(n*q - n + 1)) // q
    f = modele.addVars(m_n, vtype=GRB.INTEGER, lb=f_inf, ub=f_sup, name="f")

    # Posons maintenant les contraintes, soient les équations auxquelles
    # notre modèle est sujet. Comme on a déjà posé les bornes sur nos para-<
    # mètres, il ne suffit que d'ajouter les m-n égalités.
    for i in range(m_n):
        cote_gauche_eq = gp.quicksum(W[i, j] * x[j] for j in range(n)) - x[n+i] + q * f[i]
        cote_droit_eq = -u[i]
        modele.addConstr(cote_gauche_eq == cote_droit_eq, name=f"eq_{i}")

    # Terminons notre modèle en y ajoutant la fonction à minimiser.
    modele.setObjective(gp.quicksum(x[k]*x[k] for k in range (m)), GRB.MINIMIZE)
    #modele.setParam('OutputFlag', 0)
    modele.optimize()

    nodes = modele.NodeCount
    runtime = modele.Runtime

    # Selon le statut du modèle à la fin de son exécution, on retourne 
    # soit la solution soit rien.
    statuts_valides = [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]
    if modele.status in statuts_valides and modele.SolCount > 0:
        # On extrait la sol e du vecteur de variable de decision x
        e_sol = np.array([int(x[k].X) for k in range(m)])
        
        # On isole la partie e0 pour retrouver s, le secret,
        # via le calcul présenté en remarque 13 et
        # dans le théorème de réduction
        e0 = e_sol[:n]
        s_hat = (A_0inv @ (b_0 - e0)) % q
        
        # et on retourne le tout
        return s_hat, e_sol, nodes, runtime
    
    # Autrement, si rien n'a ete trouve:
    return None, None, nodes, runtime

# Nous proposons ici une méthode alternative de résoution du problème.
# 
def solve_lwe_centrage_mod(A, b, t, q, timelimit = 1300):

    return 

# Cette fonction réutilises le solveur que nous venons de coder plus haut, mais ajoute une 
# étape de réduction LLL de la matrice W. 
def solve_lwe_lll(A, b, q, t, time_limit=1300):
    
    # Tout est identique à la fonction précédente pour commencer.
    m, n = A.shape
    m_n = m - n 
    A_0 = A[0:n, :]
    A_1 = A[n:m, :]
    b_0 = b[:n]
    b_1 = b[n:]
    A_0inv = np.array(Matrix(A_0).inv_mod(q)).astype(int)
    W = (A_1 @ A_0inv) % q
    u = (b_1 - W @ b_0) % q


    
    # 1. Centrage modulaire (Le plus efficace pour Gurobi)
    # Ramène les coefficients de [0, q-1] vers [-q/2, q/2]
    W = np.where(W > q//2, W - q, W)
    u = np.where(u > q//2, u - q, u)

    # 2. Réduction LLL (Préconditionnement du réseau des contraintes)
    # On empile W transposé et q*I pour que LLL comprenne le modulo
    M_lat = np.vstack((W.T, q * np.eye(m_n, dtype=int)))
    M_fpylll = IntegerMatrix.from_matrix(M_lat.astype(int).tolist())
    
    # LLL réduit la matrice sur place
    LLL.reduction(M_fpylll)
    
    # On extrait la nouvelle base réduite en forçant la lecture C++ vers Python
    nrows = M_fpylll.nrows
    ncols = M_fpylll.ncols
    M_red = np.array([[M_fpylll[i, j] for j in range(ncols)] for i in range(nrows)])
    
    W_red = M_red[:n, :].T
    
    # NOTE: Pour ce script, on garde le W centré classique pour Gurobi.
    # Utiliser W_red nécessiterait de recalculer les bornes [-t, t] 
    # car les variables x_i seraient combinées.
    # ---------------------------------------------------------


    # Le reste est encore une fois identique.
    modele = gp.Model("SolveurLWE")
    modele.setParam('TimeLimit', time_limit)
    modele.setParam('SolutionLimit', 1)

    # Posons les variables de décision, soit les x_k qui sont bornés par t
    x = modele.addVars(m, vtype=GRB.INTEGER, lb=-t, ub=t, name="x")

    # Et les variables f alternatives pour le modulo, pour 
    # lesquelles ont doit calculer leurs bornes.
    f_inf = -1 * ((t*(n*q - n + 1) + q - 1) // q)
    f_sup = (t*(n*q - n + 1)) // q
    f = modele.addVars(m_n, vtype=GRB.INTEGER, lb=f_inf, ub=f_sup, name="f")

    # Posons maintenant les contraintes
    for i in range(m_n):
        # On utilise le W centré qui préserve l'indépendance des variables
        cote_gauche_eq = gp.quicksum(W[i, j] * x[j] for j in range(n)) - x[n+i] + q * f[i]
        cote_droit_eq = -u[i]
        modele.addConstr(cote_gauche_eq == cote_droit_eq, name=f"eq_{i}")

    # Terminons notre modèle en y ajoutant la fonction à minimiser.
    modele.setObjective(gp.quicksum(x[k]*x[k] for k in range (m)), GRB.MINIMIZE)
    #modele.setParam('OutputFlag', 0)
    modele.optimize()

    nodes = modele.NodeCount
    runtime = modele.Runtime

    # Selon le statut du modèle à la fin de son exécution
    statuts_valides = [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]
    if modele.status in statuts_valides and modele.SolCount > 0:
        # On extrait la sol e du vecteur de variable de decision x
        e_sol = np.array([int(x[k].X) for k in range(m)])
        
        # On isole la partie e0 pour retrouver s, le secret
        e0 = e_sol[:n]
        s_hat = (A_0inv @ (b_0 - e0)) % q
        
        # et on retourne le tout
        return s_hat, e_sol, nodes, runtime
    
    # Autrement, si rien n'a ete trouve:
    return None, None, nodes, runtime

#
def solve_lwe_centre(A, b, t, q, timelimit = 1300):

    return 
