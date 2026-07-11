import numpy as np
import pandas as pd

from gen_instance import gen_instance
from test_auto import verifier_solution
from solveur import solve_lwe
from gen_parametres import *

def test_style_instance(func, nom_test, config, nb_tests=3):
    m, n, q, t = func(**config)

    runtimes = []
    nodes = []
    succces = []

    for _ in range(nb_tests):
        try:
            A, b, q ,t, s, e = gen_instance(m, n, q, t)
            s_hat, e_hat, nodes_exp, runtime = solve_lwe(A, b, q, t)

            if s_hat is not None:
                valide = verifier_solution(s, e, s_hat, e_hat)
                runtimes.append(runtime)
                nodes.append(nodes_exp)
                succces.append(valide)
            else:
                runtimes.append(runtime)
                nodes.append(nodes_exp)
                succces.append(False)

        except ValueError:
            print("A_0 pas inversible et donc ignoree")
            continue
    
    if not runtimes:
        return {"experience": nom_test, "m": m, "n": n, "q": q, "t": t, 
                "runtime": None, "nodes_count": None, "statut": "erreur_generation"}
    
    tous_bons = all(succces)
    stat_runtime = np.mean(runtimes) if tous_bons else np.max(runtimes)
    stat_nodes = np.mean(nodes) if tous_bons else np.max(nodes)
    statut = "resolu" if tous_bons else "non_resolu"

    return {
        "experience": nom_test,
        "m": m,
        "n": n,
        "q": q,
        "t": t,
        "runtime": round(stat_runtime, 4), 
        "nodes_count": int(np.round(stat_nodes)), 
        "statut": statut
    }