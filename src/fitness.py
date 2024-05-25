from typing import List
import numpy as np
from .utils import TravelerMap
from numba.experimental import jitclass 
from numba import float32, int32    # import the types
from numba.typed import List as NumbaList

spec = [
    ('best_result', float32),
    ('best_solution',  int32[:]),
    ('result_age', int32),
    ('distances', float32[:,:]),
    ('dim', int32)
]

@jitclass(spec)
class TravelerFitness:
    def __init__(self, dim: int, distances: List[List[float]]) -> None:
        self.best_result = 100000000
        self.best_solution = np.zeros(dim, dtype=np.int32)
        self.result_age = 0
        self.distances = distances
        self.dim = dim
        
    def calculate_fitness(self, solution: List[int]) -> float:
        cumulative_path_len = 0

        cumulative_path_len += self.distances[0][solution[0]]

        for node_idx in range(len(solution)-1):
            cumulative_path_len += self.distances[solution[node_idx]][solution[node_idx+1]]

        cumulative_path_len += self.distances[solution[-1]][0]

        if self.best_result > cumulative_path_len:
            self.best_result = cumulative_path_len
            self.best_solution = np.copy(solution)
            self.result_age = 0

        return cumulative_path_len
