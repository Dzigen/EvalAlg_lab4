import random
from typing import List
import numpy as np

from numba.experimental import jitclass 
from numba import int32
import numba

spec = [
    ('cand_d', int32),
    ('genes_idx', int32[:]),
    ('pairs_num', int32)
]

@jitclass(spec)
class TravelerCrossover:
    
    def __init__(self, dimension: int, crossover_pairs: int) -> None:
        self.cand_d = dimension
        self.genes_idx = np.arange(1, self.cand_d-1, dtype=np.int32)
        self.pairs_num = crossover_pairs
        
    def mate(self, population: List[List[int]]) -> List[List[int]]:
        new_solutions = np.zeros(shape=(self.pairs_num*2, self.cand_d), dtype=np.int32)
        pop_idxs = np.arange(len(population),dtype=np.int32)

        for i in range(self.pairs_num):
            #
            idx1, idx2 = np.random.choice(pop_idxs, size=2, replace=False)
            sol1, sol2 = population[idx1], population[idx2]
            delim_idx = np.random.choice(self.genes_idx, size=1, replace=False)[0]
            
            #
            new_sol1 = self.create_new_sol(delim_idx, sol1, sol2)
            new_sol2 = self.create_new_sol(delim_idx, sol2, sol1)

            new_solutions[i*2] = new_sol1
            new_solutions[(i*2)+1] = new_sol2

        return new_solutions
    
    def create_new_sol(self, delim_idx: int, sol1: List[int], sol2: List[int]) -> List[int]:
        new_sol = np.zeros(shape=self.cand_d, dtype=np.int32) 
        new_sol[:delim_idx] = sol1[:delim_idx]

        sol2_part = np.array([gene for gene in sol2[delim_idx:] if gene not in new_sol])
        new_sol[delim_idx:delim_idx + len(sol2_part)] = sol2_part

        sol1_part = np.array([gene for gene in sol1[delim_idx:] if gene not in new_sol])
        new_sol[delim_idx + len(sol2_part):delim_idx + len(sol2_part) + len(sol1_part)] = sol1_part

        return new_sol