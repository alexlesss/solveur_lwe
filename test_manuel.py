import numpy as np

from solveurs import solve_lwe 

if __name__ == "__main__":
    # 
    q = 17
    t = 1
    n = 4
    m = 6

    # La matrice A publique de l'exemple
    A = np.array([
        [14, 15,  5,  2],
        [13, 14, 14,  6],
        [ 6, 10, 13,  1],
        [10,  4, 12, 16],
        [ 9,  5,  9,  6],
        [ 3,  6,  4,  5],
        [ 6,  7, 16,  2]
    ])

    # ainsi que le vecteur public de l'exemple
    b = np.array([8, 16, 3, 12, 9, 16, 3])

    s_reel = np.array([0, 13, 9, 11])

    print(f"Paramètres : n={n}, m={m}, q={q}, t={t}")
    print("Secret attendu (s) :", s_reel)
    print("-" * 30)

    print("\nLancement de Gurobi...")
    s_trouve, e_trouve, nodes, runtime = solve_lwe(A, b, q, t, time_limit=60)

    if s_trouve is not None:
        print("Secret trouvé (s) :", s_trouve)
        
        # Test final
        if np.array_equal(s_reel, s_trouve):
            print("\n Resultat identique")
        else:
            print("\n Solution trouvee mais pas bonne")
    else:
        print("\n Le solveur n'a rien trouvé.")