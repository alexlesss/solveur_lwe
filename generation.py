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
    n = 11
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

# La premiere est avec la gaussienne troquee, plus theorique. 
def gen_instance_gauss(m, n, q, t, seed):
    rng = np.random.default_rng(seed)

    # Generation des parametres A et s 
    A = rng.integers(0, q, size=(m, n))
    s = rng.integers(0, q, size=n)

    # Generation de e selon la gaussienne tronquee
    sigma = t / 3.0
    while True:
        e = np.round(rng.normal(0, sigma, size= m)).astype(int)
        if np.max(np.abs(e)) <= t:
            break

    # Calcul de b
    b = (A @ s + e) % q 

    return A, b, q, t, s, e


# La seconde est avec la distribution binomiale centree.
# Plus pratique, plus style Kyber.
def gen_instance_cbd(m, n, q, t, seed):
    rng = np.random.default_rng(seed)

    A = rng.integers(0, q, size=(m, n))
    s = rng.integers(0, q, size=n)

    # Generation de e selon la CBD (Centered Binomial Distribution)
    # a et b maximum a 2 exclus, ca nous fait deux matrices de taille m x t 
    bits_a = rng.integers(0, 2, size=(m, t))
    bits_b = rng.integers(0, 2, size=(m, t))
    
    # en combinant ces deux matrices, on genere e
    e = np.sum(bits_a, axis=1) - np.sum(bits_b, axis=1)

    b = (A @ s + e) % q 
    
    return A, b, q, t, s, e


