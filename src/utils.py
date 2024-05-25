import math
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

class TravelerMap:
    def __init__(self) -> None:
        pass

    def load_map(self, file_path: str) -> None:
        with open(file_path, 'r', encoding='utf-8') as fd:
            data = list(map(lambda item: item.strip(), fd.readlines()))
        self.nodes_amount = int(data[5].split()[2])

        raw_nodes = data[8:-1]
        self.nodes = self.extract_nodes(raw_nodes)
        self.distances = np.array(self.compute_nodes_distances(self.nodes), dtype=np.float32)


    def extract_nodes(self, raw_nodes: List[str]) -> List[List[int]]:
        nodes = []
        for item in raw_nodes:
            _, x, y = item.split()
            nodes.append([int(x), int(y)])

        return nodes
    
    def compute_nodes_distances(self, nodes: List[List[int]]) -> List[List[float]]:
        distances = []
        for main_node in nodes:
            main_node_dists: List[float] = []
            for node in nodes:
                dist: float = math.sqrt(pow(node[0] - main_node[0], 2) + pow(node[1] - main_node[1], 2))
                main_node_dists.append(round(dist, 3))
            distances.append(main_node_dists)

        return distances

    def get_distance(self, node_idx1: int, node_idx2: int) -> float:
        return self.distances[node_idx1][node_idx2]
    
    def plot_solution(self, solution, c='black', mec='r', mfc='r', ms=3):
        x = list(map(lambda node_idx: self.nodes[node_idx][0], solution))
        x.append(self.nodes[solution[0]][0])
        y = list(map(lambda node_idx: self.nodes[node_idx][1], solution))
        y.append(self.nodes[solution[0]][1])

        plt.plot(x, y, marker='o', c=c, mec=mec, mfc=mfc, ms=ms)
        plt.axis('off')
        plt.show()