import random
import os
import heapq
import json

import numpy as np

from dataclasses import dataclass
from typing import Dict, List
from tqdm import tqdm


class Node:
    def __init__(self, name: str, point: List[float], level: int):
        self.name = name
        self.point = point
        self.level = level
        # Using dictionaries to store neighbors with names as keys
        self.neighbors = {i: {} for i in range(level + 1)}

class HNSWGraph:
    def __init__(self, dim, max_elements, M=50, ef=10, level=3, dict_path=None):
        """
        Initializes the HNSW graph with given parameters.
        """
        self.dim = dim
        self.max_elements = max_elements
        self.M = M
        self.ef = ef
        self.nodes = {}
        self.entry_point = None
        if dict_path:
            self._initialize_from_dict(dict_path)

    def _initialize_from_dict(self, dict_path):
        with open(dict_path, 'r') as f:
            data = json.load(f)
        for category, info in tqdm(data.items()):
            centroid = info['centroid']
            if 'level' in info.keys():
                level = info['level']
            else:
                level = None
            self.add_node(category, centroid, level)

    def add_node(self, name, point, level=None):
        if level is None:
            level = self._random_level()
        new_node = Node(name, point, level)
        if not self.nodes:
            self.entry_point = new_node
        else:
            self._add_to_graph(new_node)
        self.nodes[name] = new_node

    def _random_level(self):
        return np.random.geometric(0.5) - 1

    def _add_to_graph(self, new_node):
        curr = self.entry_point
        level = curr.level

        for l in range(level, new_node.level, -1):
            curr = self._greedy_search(curr, new_node.point, l)

        for l in range(min(new_node.level, level), -1, -1):
            neighbors = self._get_nearest_neighbors(curr, new_node.point, l)
            new_node.neighbors[l] = {n.name: n for n in neighbors}
            for neighbor in neighbors:
                neighbor.neighbors[l][new_node.name] = new_node

    def _greedy_search(self, curr, point, level):
        closest = curr
        closest_dist = self._euclidean_dist(curr.point, point)
        while True:
            found_closer = False
            for neighbor in closest.neighbors[level].values():
                dist = self._euclidean_dist(neighbor.point, point)
                if dist < closest_dist:
                    closest = neighbor
                    closest_dist = dist
                    found_closer = True
            if not found_closer:
                break
        return closest

    def _get_nearest_neighbors(self, curr, point, level):
        visited = set()
        visited.add(curr)
        # Use node name as a secondary sort key to avoid TypeError when distances are equal
        candidates = [(self._euclidean_dist(curr.point, point), curr.name, curr)]
        nearest_neighbors = []

        while candidates:
            dist, _, node = heapq.heappop(candidates)  # Include the name in unpacking but don't use it further
            if len(nearest_neighbors) >= self.M:
                break
            if node not in nearest_neighbors:
                nearest_neighbors.append(node)
            # Expand the search to the neighbors of the current node
            for neighbor in node.neighbors[level].values():
                if neighbor not in visited:
                    visited.add(neighbor)
                    # Include node name to prevent TypeError due to identical distances
                    heapq.heappush(candidates, (self._euclidean_dist(neighbor.point, point), neighbor.name, neighbor))

        return nearest_neighbors



    def _euclidean_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - p2)

    def knn_query(self, query_point, k):
        curr = self.entry_point
        level = curr.level

        for l in range(level, -1, -1):
            curr = self._greedy_search(curr, query_point, l)

        candidates = [(self._euclidean_dist(curr.point, query_point), curr)]
        visited = set()
        visited.add(curr)
        nearest_neighbors = []

        while candidates and len(nearest_neighbors) < k:
            dist, node = heapq.heappop(candidates)
            nearest_neighbors.append((dist, node))
            for neighbor in node.neighbors[0].values():
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(candidates, (self._euclidean_dist(neighbor.point, query_point), neighbor))

        return nearest_neighbors
    
    def save_model(self, path):
        data = {name: {'centroid': node.point, 'level': node.level} for name, node in self.nodes.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    # Example Usage
    dim = 1500  # Dimension of the feature space
    num_elements = 100  # Total number of elements to add to the graph

    # Initialize the HNSW graph with specified dimensions and number of elements
    hnsw = HNSWGraph(dim, num_elements)

    # Add random points to the graph
    for i in range(num_elements):
        point = np.random.rand(dim).tolist()  # Generate a random vector of size 'dim'
        hnsw.add_node(f"random_{i}", point)  # Add node with a unique name and random point

    # Query the graph
    query_point = np.random.rand(dim).tolist()  # Generate a random query point
    nearest_neighbors = hnsw.knn_query(query_point, 10)  # Retrieve the 10 nearest neighbors to the query point
    print(type(nearest_neighbors[0][1]))  # Print the type of the nearest neighbors
    print("Nearest neighbors:", [neighbor[1].name for neighbor in nearest_neighbors])  # Print the names of the nearest neighbors