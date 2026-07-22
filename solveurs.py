import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sympy import Matrix
from fpylll import IntegerMatrix, LLL

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
def solve_lwe(A,b,q,t, time_limit=300):
    
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
    modele.setParam('TimeLimit', 300)
    # Comme les problèmes sur lesquels nous allons travailler seront bien con-
    # ditionnés (nous présenterons ce conditionnement plus tard), nous pouvons
    # assumer qu'il n'y a qu'une seule solution valide.
    modele.setParam('SolutionLimit', 1)
    modele.setParam('Threads', 6)

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

# Cette fonction réutilises le solveur que nous venons de coder plus haut, mais ajoute une 
# mais on ajoute le centrage modulaire. 
def solve_lwe_cmod(A, b, q, t, time_limit=300):
    
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

    # 1. Centrage modulaire 
    # Ramène les coefficients de [0, q-1] vers [-q/2, q/2]
    W = np.where(W > q//2, W - q, W)
    u = np.where(u > q//2, u - q, u)

    # Le reste est encore une fois identique.
    modele = gp.Model("SolveurLWE")
    modele.setParam('TimeLimit', time_limit)
    modele.setParam('SolutionLimit', 1)
    modele.setParam('Threads', 6)

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


def solve_lwe_shirase_lll(A, b, q, t, time_limit=300):
    m, n = A.shape
    m_n = m - n 
    
    A_0 = A[0:n, :]
    A_1 = A[n:m, :]
    b_0 = b[:n]
    b_1 = b[n:]
    
    A_0inv = np.array(Matrix(A_0).inv_mod(q)).astype(int)
    W = (A_1 @ A_0inv) % q
    u = (b_1 - W @ b_0) % q
    
    # 1. Centrage modulaire (Crucial pour Shirase)
    W = np.where(W > q//2, W - q, W)
    u = np.where(u > q//2, u - q, u)

    # ---------------------------------------------------------
    # 2. RÉDUCTION LLL ADAPTÉE POUR SHIRASE
    # ---------------------------------------------------------
    # On réduit W transposé et on extrait la matrice unimodulaire T
    # telle que M_red = T * M_fpylll
    M_fpylll = IntegerMatrix.from_matrix(W.T.astype(int).tolist())
    T_fpylll = IntegerMatrix.identity(n)
    
    LLL.reduction(M_fpylll, T_fpylll)
    
    # Extraction vers NumPy
    W_red_T = np.array([[M_fpylll[i, j] for j in range(m_n)] for i in range(n)])
    T_mat = np.array([[T_fpylll[i, j] for j in range(n)] for i in range(n)])
    
    # On a : W_red_T = T_mat @ W.T
    # En transposant : W_red = W @ T_mat.T
    W_red = W_red_T.T
    U = T_mat.T # U est notre matrice de passage telle que W_red = W @ U

    # ---------------------------------------------------------
    # 3. MODÈLE GUROBI (Strictement Shirase)
    # ---------------------------------------------------------
    modele = gp.Model("SolveurLWE_Shirase_LLL")
    modele.setParam('TimeLimit', time_limit)
    modele.setParam('SolutionLimit', 1)
    modele.setParam('Threads', 6)

    # Tes variables originales x, avec leurs vraies bornes [-t, t]
    x = modele.addVars(m, vtype=GRB.INTEGER, lb=-t, ub=t, name="x")

    # Tes variables f de modulo
    f_inf = -1 * ((t*(n*q - n + 1) + q - 1) // q)
    f_sup = (t*(n*q - n + 1)) // q
    f = modele.addVars(m_n, vtype=GRB.INTEGER, lb=f_inf, ub=f_sup, name="f")

    # LA MAGIE ICI : On crée des variables z (unbounded) pour utiliser W_red
    z = modele.addVars(n, vtype=GRB.INTEGER, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z")

    # Contraintes de liaison : x0 = U * z. 
    # Gurobi va utiliser ces équations pour borner 'z' intelligemment.
    for i in range(n):
        modele.addConstr(
            x[i] == gp.quicksum(int(U[i, j]) * z[j] for j in range(n)), 
            name=f"link_x0_z_{i}"
        )

    # Les contraintes principales de Shirase, mais avec W_red et z !
    # W_red * z - x1 + q * f = -u
    for i in range(m_n):
        cote_gauche = gp.quicksum(int(W_red[i, j]) * z[j] for j in range(n)) - x[n+i] + q * f[i]
        modele.addConstr(cote_gauche == int(-u[i]), name=f"eq_shirase_{i}")

    # Fonction objectif inchangée
    modele.setObjective(gp.quicksum(x[k]*x[k] for k in range(m)), GRB.MINIMIZE)
    modele.optimize()

    # ---------------------------------------------------------
    # 4. EXTRACTION (inchangée)
    # ---------------------------------------------------------
    nodes = modele.NodeCount
    runtime = modele.Runtime

    statuts_valides = [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]
    if modele.status in statuts_valides and modele.SolCount > 0:
        # round() pour éviter les 0.9999 de Gurobi
        e_sol = np.array([int(round(x[k].X)) for k in range(m)])
        
        e0 = e_sol[:n]
        s_hat = (A_0inv @ (b_0 - e0)) % q
        
        return s_hat, e_sol, nodes, runtime
    
    return None, None, nodes, runtime 


def solve_lwe_bdd(A, b, q, t, time_limit=300):
    m, n = A.shape
    m_n = m - n 
    
    A_0 = A[0:n, :]
    A_1 = A[n:m, :]
    b_0 = b[:n]
    b_1 = b[n:]
    
    # Calcul de W et centrage
    A_0inv = np.array(Matrix(A_0).inv_mod(q)).astype(int)
    W = (A_1 @ A_0inv) % q
    u = (b_1 - W @ b_0) % q
    
    W = np.where(W > q//2, W - q, W)
    u = np.where(u > q//2, u - q, u)

    # ---------------------------------------------------------
    # 1. CONSTRUCTION DU RÉSEAU PRIMAL ET RÉDUCTION LLL
    # ---------------------------------------------------------
    # On construit la matrice B de dimension (m x m) dont les colonnes 
    # génèrent l'espace des solutions valides.
    # Pour fpylll, on doit empiler ça en lignes, donc on crée B_T (B transposée).
    B_T = np.block([
        [np.eye(n, dtype=int), W.T],
        [np.zeros((m_n, n), dtype=int), q * np.eye(m_n, dtype=int)]
    ])
    
    # Réduction LLL sur les lignes
    M_fpylll = IntegerMatrix.from_matrix(B_T.tolist())
    LLL.reduction(M_fpylll)
    
    # On extrait la nouvelle base et on la re-transpose pour l'avoir en colonnes
    B_red_T = np.array([[M_fpylll[i, j] for j in range(m)] for i in range(m)])
    B_red = B_red_T.T
    
    # On prépare le vecteur cible u_bar (de taille m)
    u_bar = np.concatenate((np.zeros(n, dtype=int), u))

    # ---------------------------------------------------------
    # 2. MODÈLE GUROBI
    # ---------------------------------------------------------
    modele = gp.Model("SolveurLWE_LLL")
    modele.setParam('TimeLimit', time_limit)
    modele.setParam('SolutionLimit', 1)
    modele.setParam('Threads', 6)

    # Les variables d'erreur e (anciennement tes x), bornées par l'erreur LWE [-t, t]
    e = modele.addVars(m, vtype=GRB.INTEGER, lb=-t, ub=t, name="e")

    # Les variables y: ce sont les coefficients dans notre NOUVELLE base LLL
    # On laisse les bornes infinies, le presolve de Gurobi va les déduire
    # à partir des bornes strictes de 'e'.
    y = modele.addVars(m, vtype=GRB.INTEGER, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")

    # Contraintes : on force le vecteur e à appartenir au réseau shifté
    # e = B_red * y + u_bar
    for i in range(m):
        modele.addConstr(
            e[i] == gp.quicksum(int(B_red[i, j]) * y[j] for j in range(m)) + int(u_bar[i]),
            name=f"lat_eq_{i}"
        )

    # Objectif : Minimiser la norme de l'erreur
    modele.setObjective(gp.quicksum(e[k]*e[k] for k in range(m)), GRB.MINIMIZE)
    modele.optimize()

    # ---------------------------------------------------------
    # 3. EXTRACTION
    # ---------------------------------------------------------
    nodes = modele.NodeCount
    runtime = modele.Runtime

    statuts_valides = [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]
    if modele.status in statuts_valides and modele.SolCount > 0:
        # round() évite les erreurs de précision flottante de Gurobi (ex: 0.999999)
        e_sol = np.array([int(round(e[k].X)) for k in range(m)])
        
        e0 = e_sol[:n]
        s_hat = (A_0inv @ (b_0 - e0)) % q
        
        return s_hat, e_sol, nodes, runtime
    
    return None, None, nodes, runtime


