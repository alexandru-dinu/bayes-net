from collections import defaultdict
from utils import get_sorted_edges, add_weighted_undir_edge


parents = {}
ranks = {}


class DisjointSets:
    def __init__(self):
        self.parents = {}
        self.ranks = {}

    def make_set(self, v):
        self.parents[v] = v
        self.ranks[v] = 0

    def find_set(self, v):
        if self.parents[v] != v:
            self.parents[v] = self.find_set(self.parents[v])
        return self.parents[v]

    def union(self, v1, v2):
        x = self.find_set(v1)
        y = self.find_set(v2)

        if x == y:
            return

        if self.ranks[x] < self.ranks[y]:
            self.parents[x] = y
        else:
            self.parents[y] = x

        if self.ranks[x] == self.ranks[y]:
            self.ranks[y] += 1


def kruskal(graph):
    """
    graph = {
        node_i: {
            node_j: w_ij
        }
    }
    """
    ds = DisjointSets()

    mst = defaultdict(lambda: {})
    s_edges = get_sorted_edges(graph, reverse=True)

    for v in graph.keys():
        ds.make_set(v)

    for (u, v, w) in s_edges:
        if ds.find_set(u) != ds.find_set(v):
            ds.union(u, v)
            add_weighted_undir_edge(mst, u, v, w)

    return mst
