import os
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

if __name__ == "__main__":
    print("==================================================")
    print("DÉMARRAGE DU BENCHMARK POUR LWE")
    print("==================================================")
    
    NOMBRE_DE_TESTS = 2

    # Valeurs itérées
    valeurs_m = [15, 20, 25, 30, 35, 40] 
    # Valeurs fixes (au besoin, si pas deja fixew)
    
    for m_val in valeurs_m:
        # Préparation des arguments selon le type de test
        config = {"m": m_val}
        label = f"m_mobile (m={m_val})"
        
        print(f"Lancement de {label}...")
        
        # On roule le test et on accumule la ligne de stats
        ligne_resultat = test_style_instance(
            func=m_mobile,
            nom_test=label,  
            config=config, 
            nb_tests=NOMBRE_DE_TESTS
        )
        
        print(f"  -> Terminé. t calculé = {ligne_resultat['t']} | Runtime moyen = {ligne_resultat['runtime']}s | Statut = {ligne_resultat['statut']} | Nombre de nodes = {ligne_resultat['nodes_count']}\n")
        
        df_ligne = pd.DataFrame([ligne_resultat])
        nom_fichier = "resultats_bench.csv"
        df_ligne.to_csv(nom_fichier, mode='a', header=not os.path.exists(nom_fichier), index=False)
    
    print("==================================================")
    print(f"BENCHMARK TERMINÉ!")
