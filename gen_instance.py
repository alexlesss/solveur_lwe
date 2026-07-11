# Ce fichier prend en parametres les valeurs m, n, q, t valides générés dans un générateur 
# de parametres quelconque et retourne une instance LWE qui utilise ces parametres

import numpy as np

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
