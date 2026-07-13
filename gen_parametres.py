import math
from sympy import nextprime


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
    q = nextprime(n**2)
    t = 3
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

