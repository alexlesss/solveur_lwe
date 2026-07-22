import os
import numpy as np
import pandas as pd
from test_auto import verifier_solution
from solveurs import *
from generation import *

# Notre fichier de benchmark est malheureusement manuel, il pourrait etre interessant de creer une interface.
def test_instance(func, nom_test, config, seeds):
    m, n, q, t = func(**config)
    
    resultats_detailles = []
    
    runtimes = []
    nodes = []
    succces = []

    for seed in seeds:
        try:
            A, b, q, t, s, e = gen_instance_cbd(m, n, q, t, seed)
            s_hat, e_hat, nodes_exp, runtime = solve_lwe(A, b, q, t)

            if s_hat is not None:
                valide = verifier_solution(s, e, s_hat, e_hat)
            else:
                valide = False
            
            statut = "resolu" if valide else "non_resolu"
            
            resultats_detailles.append({
                "experience": nom_test,
                "m": m,
                "n": n,
                "q": q,
                "t": t,
                "seed": str(seed),  
                "runtime": round(runtime, 4), 
                "nodes_count": int(nodes_exp), 
                "statut": statut
            })
            
            runtimes.append(runtime)
            nodes.append(nodes_exp)
            succces.append(valide)

        except ValueError:
            print(f"Seed {seed} : A_0 pas inversible et donc ignorée")
            resultats_detailles.append({
                "experience": nom_test, "m": m, "n": n, "q": q, "t": t, 
                "seed": str(seed), "runtime": None, "nodes_count": None, "statut": "erreur_generation"
            })
            continue
            
    if runtimes: 
        tous_bons = all(succces)
        stat_runtime = np.mean(runtimes) if tous_bons else np.max(runtimes)
        stat_nodes = np.mean(nodes) if tous_bons else np.max(nodes)
        statut_moyen = "resolu" if tous_bons else "non_resolu"
        
        resultats_detailles.append({
            "experience": nom_test,
            "m": m,
            "n": n,
            "q": q,
            "t": t,
            "seed": "MOYENNE",
            "runtime": round(stat_runtime, 4), 
            "nodes_count": int(np.round(stat_nodes)), 
            "statut": statut_moyen
        })
            
    return resultats_detailles

if __name__ == "__main__":
    print("==================================================")
    print("DÉMARRAGE DU BENCHMARK POUR LWE")
    print("==================================================")
    
    SEEDS_A_TESTER = [42, 123, 999]
    # si on souhaite de l'alea, simplement decommenter la ligne suivante et commenter la precedente
    # SEEDS_A_TESTER = [np.random.randint(0, 10000) for _ in range(3)]


    valeurs_m = [20, 25, 30, 35] 
    
    nom_fichier = "resultats_bench.csv"

    for m_val in valeurs_m:
        config = {"m": m_val}
        label = f"m_mobile (m={m_val})"
        
        print(f"Lancement de {label}...")
        
        lignes_resultats = test_instance(
            func=m_mobile,
            nom_test=label,  
            config=config, 
            seeds=SEEDS_A_TESTER  
        )
        
        # Affichage structuré dans le terminal
        for ligne in lignes_resultats:
            if ligne['seed'] == "MOYENNE":
                print(f"  => BILAN GLOBAL : Runtime moyen = {ligne['runtime']}s | Statut = {ligne['statut']} | Nodes = {ligne['nodes_count']}\n")
            else:
                print(f"  -> Seed {ligne['seed']} | Runtime = {ligne['runtime']}s | Statut = {ligne['statut']} | Nodes = {ligne['nodes_count']}")
        
        # Sauvegarde de tout le bloc (incluant la ligne moyenne) dans le CSV
        df_lignes = pd.DataFrame(lignes_resultats)
        df_lignes.to_csv(nom_fichier, mode='a', header=not os.path.exists(nom_fichier), index=False)
    
    print("==================================================")
    print("BENCHMARK TERMINÉ!")