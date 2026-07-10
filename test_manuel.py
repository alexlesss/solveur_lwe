import numpy as np

# Importe ton solveur
from solveur import solve_lwe 

if __name__ == "__main__":
    # --- 1. Paramètres hardcodés de notre exemple manuel ---
    q = 11
    t = 1
    n = 2
    m = 4

    # La matrice A publique
    A = np.array([
        [2, 5],
        [4, 1],
        [7, 3],
        [1, 8]
    ])

    # Le vecteur public b (déjà calculé modulo 11)
    b = np.array([9, 8, 8, 5])

    # La vérité cachée (ce qu'on s'attend à trouver)
    s_reel = np.array([3, 7])
    e_reel = np.array([1, 0, -1, 1])

    print("--- DÉBUT DU TEST MANUEL ---")
    print(f"Paramètres : n={n}, m={m}, q={q}, t={t}")
    print("Secret attendu (s) :", s_reel)
    print("Erreur attendue (e) :", e_reel)
    print("-" * 30)

    # --- 2. Lancement du solveur ---
    print("\nLancement de Gurobi...")
    s_trouve, e_trouve, nodes, runtime = solve_lwe(A, b, q, t, time_limit=60)

    # --- 3. Vérification ---
    if s_trouve is not None:
        print("\n✅ SOLUTION TROUVÉE PAR GUROBI :")
        print("Secret trouvé (s) :", s_trouve)
        print("Erreur trouvée (e) :", e_trouve)
        
        # Test final
        if np.array_equal(s_reel, s_trouve) and np.array_equal(e_reel, e_trouve):
            print("\n🏆 RÉSULTAT PARFAIT : Le code correspond à 100% au calcul mathématique à la main !")
        else:
            print("\n⚠️ Oups, la solution est valide mais différente du calcul attendu.")
    else:
        print("\n❌ Le solveur n'a rien trouvé.")