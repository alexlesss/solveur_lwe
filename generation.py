import math
import numpy as np
from sympy import nextprime

# IL Y A DEUX SECTIONS DANS CE FICHIER: LES PARAMETRES ET LES INSTANCES

# SECTION 1 : GENERATION DE PARAMETRES
def m_petit_n(m, n):
    q = nextprime(n**2)

    ln_numerateur = -12 * math.log(2)
    ln_denominateur = n * math.log(q)
    
    ln_fraction = ln_numerateur - ln_denominateur
    ln_terme_puissance = ln_fraction / m
    
    terme_puissance = math.exp(ln_terme_puissance)
    
    t_max_th = 0.25 * (terme_puissance * q - 1)
    t = math.floor(t_max_th)
    
    return m, n, q, max(0, t)

def m_mobile(m):
    n = 15
    q = 3329
    t = 2
    return m, n, q, max(0, t)

def q_mobile(q):
    n = 10
    m = 18
    t = 2
    return m, n, q, max(0,t)

def n_mobile(n):
    m = 20
    q = 149
    t = 2
    return m, n, q, max(0, t)

def relativement_bien_cond(m):
    n = 13
    q = nextprime(n**2)

    ln_numerateur = -7 * math.log(2)
    ln_denominateur = n * math.log(q)
    
    ln_fraction = ln_numerateur - ln_denominateur
    ln_terme_puissance = ln_fraction / m
    
    terme_puissance = math.exp(ln_terme_puissance)
    
    t_max_th = 0.25 * (terme_puissance * q - 1)
    t = math.floor(t_max_th)
    
    return m, n, q, max(0, t)



# SECTION 2 - GENERATION D'INSTANCES 
# Ces generateurs d'instances utilisent les valeurs retournees par un generateur
# de parametres quelconque et retourne une instance LWE qui utilise ces parametres

def gen_instance(m, n, q, t):

    # Generation des parametres A et s aléatoirement
    A = np.random.randint(0, q, size=(m, n))
    s = np.random.randint(0, q, size=n)

    # Generation de e selon la gaussienne tronquee
    sigma = t / 3.0
    while True:
        e = np.round(np.random.normal(0, sigma, size= m)).astype(int)
        if np.max(np.abs(e)) <= t:
            break

    # Calcul de b
    b = (A @ s + e) % q 

    return A, b, q, t, s, e 

def gen_instance_fixe(m, n, q, t):
    #meme code que plus haut, mais on utilise un seed fixe
    np.random.seed(42)

    # Generation des parametres A et s (maintenant 100% déterministes)
    A = np.random.randint(0, q, size=(m, n))
    s = np.random.randint(0, q, size=n)

    # Generation de e selon la gaussienne tronquee
    sigma = t / 3.0
    while True:
        e = np.round(np.random.normal(0, sigma, size= m)).astype(int)
        if np.max(np.abs(e)) <= t:
            break

    # Calcul de b
    b = (A @ s + e) % q 
    
    np.random.seed(None)

    return A, b, q, t, s, e

def gen_instance_cbd(m, n, q, t):
    # Comme gen instance mais via cbd
    A = np.random.randint(0, q, size=(m, n))
    s = np.random.randint(0, q, size=n)

    # Generation de e selon la CBD (Centered Binomial Distribution)
    # a et b maximum a 2 exclus, ca nous fait deux matrices de taille m x t 
    bits_a = np.random.randint(0, 2, size=(m, t))
    bits_b = np.random.randint(0, 2, size=(m, t))
    
    # en combinant ces deux matrices, on genere e
    e = np.sum(bits_a, axis=1) - np.sum(bits_b, axis=1)

    # Calcul de b
    b = (A @ s + e) % q 
    
    return A, b, q, t, s, e

def gen_instance_fixe_cbd(m, n, q, t):
    #meme code que gen instance fixe mais avec cbd
    np.random.seed(42)

    A = np.random.randint(0, q, size=(m, n))
    s = np.random.randint(0, q, size=n)

    # Generation de e selon la CBD (Centered Binomial Distribution)
    # a et b maximum a 2 exclus, ca nous fait deux matrices de taille m x t 
    bits_a = np.random.randint(0, 2, size=(m, t))
    bits_b = np.random.randint(0, 2, size=(m, t))
    
    # en combinant ces deux matrices, on genere e
    e = np.sum(bits_a, axis=1) - np.sum(bits_b, axis=1)

    b = (A @ s + e) % q 
    np.random.seed(None)
    return A, b, q, t, s, e


