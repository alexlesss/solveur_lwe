import numpy as np

def verifier_solution(s, e, s_gurobi, e_gurobi):
    if s_gurobi is None or e_gurobi is None:
        return False
        
    bon_s = np.array_equal(s, s_gurobi)
    bon_e = np.array_equal(e, e_gurobi)
    
    return bon_s and bon_e