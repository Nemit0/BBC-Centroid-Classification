import random
import os
import heapq

import numpy as np

from dataclasses import dataclass


class Node:
    def __init__(self, point, level):
        self.point = point
        self.level = level
        self.neighbors = [[] for _ in range(level + 1)]

class HNSWGraph:
    def __init__(self, dim, max_elements, M=16, ef=10):
        self.dim = dim
        self.max_elements = max_elements
        self.M = M
        self.ef = ef
        self.nodes = []
        self.entry_point = None

    def add_node(self, point):
        level = self._random_level()
        new_node = Node(point, level)
        if not self.nodes:
            self.entry_point = new_node
        else:
            self._add_to_graph(new_node)
        self.nodes.append(new_node)

    def _random_level(self):
        return np.random.geometric(0.5) - 1

    def _add_to_graph(self, new_node):
        curr = self.entry_point
        level = len(curr.neighbors) - 1

        for l in range(level, new_node.level, -1):
            curr = self._greedy_search(curr, new_node.point, l)

        for l in range(min(new_node.level, level), -1, -1):
            neighbors = self._get_nearest_neighbors(curr, new_node.point, l)
            new_node.neighbors[l].extend(neighbors)
            for neighbor in neighbors:
                neighbor.neighbors[l].append(new_node)

    def _greedy_search(self, curr, point, level):
        closest = curr
        closest_dist = self._euclidean_dist(curr.point, point)
        while True:
            found_closer = False
            for neighbor in closest.neighbors[level]:
                dist = self._euclidean_dist(neighbor.point, point)
                if dist < closest_dist:
                    closest = neighbor
                    closest_dist = dist
                    found_closer = True
            if not found_closer:
                break
        return closest

    def _get_nearest_neighbors(self, curr, point, level):
        candidates = [(self._euclidean_dist(curr.point, point), curr)]
        visited = set()
        visited.add(curr)
        nearest_neighbors = []

        while candidates:
            dist, node = heapq.heappop(candidates)
            nearest_neighbors.append(node)
            if len(nearest_neighbors) >= self.M:
                break
            for neighbor in node.neighbors[level]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(candidates, (self._euclidean_dist(neighbor.point, point), neighbor))

        return nearest_neighbors

    def _euclidean_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def knn_query(self, query_point, k):
        curr = self.entry_point
        level = len(curr.neighbors) - 1

        for l in range(level, -1, -1):
            curr = self._greedy_search(curr, query_point, l)

        candidates = [(self._euclidean_dist(curr.point, query_point), curr)]
        visited = set()
        visited.add(curr)
        nearest_neighbors = []

        while candidates and len(nearest_neighbors) < k:
            dist, node = heapq.heappop(candidates)
            nearest_neighbors.append((dist, node.point))
            for neighbor in node.neighbors[0]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(candidates, (self._euclidean_dist(neighbor.point, query_point), neighbor))

        return nearest_neighbors

if __name__ == "__main__":
    # Example Usage
    dim = 2
    num_elements = 100
    hnsw = HNSWGraph(dim, num_elements)

    # Add random points to the graph
    for _ in range(num_elements):
        point = np.random.rand(dim)
        hnsw.add_node(point)

    # Query the graph
    query_point = np.random.rand(dim)
    nearest_neighbors = hnsw.knn_query(query_point, 10)
    print("Nearest neighbors:", nearest_neighbors)