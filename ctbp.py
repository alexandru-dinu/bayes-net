from utils import *

from copy import copy
from itertools import combinations
from operator import itemgetter
from pprint import pprint

class BayesNet:
    def __init__(self, filename):
        self.graph = {}
        self.queries = []
        self.probabilities = {}

        lines = [l.strip() for l in open(filename).readlines()]

        for line in lines:
            n, given, probs = [a.strip().split(" ") for a in line.split(";")]
            self.graph[n[0]] = [] if given == [''] else given
            self.probabilities[n[0]] = list(map(float, probs))


    def get_prob(self, node, given=None):
        """
        p(A | B, ~C) = get_prob("A", given="10")
        """
        if given is None:
            return self.probabilities[node]
        else:
            assert len(given) == len(self.graph[node])
            return self.probabilities[node][int(given, 2)]
# -- problem


def moralize(graph):
    out = copy(graph)

    for v in get_vertices(graph):
        for (x, y) in combinations(graph[v], 2):
            print(f">>> [moralize] adding edge: ({x}, {y})")
            out[x].append(y)

    return get_undirected_graph(out)


def get_f_set(graph, node):
    edges = get_edges(graph)
    vs = get_vertices(graph)

    f_set = [
        (u1, u2) for (u1, u2) in combinations(vs, 2)
        if is_edge(graph, u1, node) and is_edge(graph, u2, node)
    ]

    return f_set


def get_induced(graph, without):
    out = {}

    for n, ns in graph.items():
        if n == without:
            continue
        out[n] = [x for x in ns if x != without]

    return out


def triangulate(graph):
    out = copy(graph)
    added_by = {}

    num_iter = len(get_vertices(graph))

    for i in range(num_iter):
        vs = get_vertices(out)
        edge_set = set(get_edges(out))

        f_sets = {u: get_f_set(out, u) for u in vs}

        candidates = {x: len(set(f_sets[x]) - edge_set) for x in vs}
        v_min = min(candidates, key=candidates.get)

        added_by[v_min] = set(f_sets[v_min]) - edge_set

        for (u1, u2) in f_sets[v_min]:
            out = add_undir_edge(out, u1, u2)

        out = get_induced(out, v_min)


    to_be_added = flatten([list(es) for es in added_by.values()])

    for (u, v) in to_be_added:
        print(f">>> [triangulate] adding edge: ({u}, {v})")
        graph = add_undir_edge(graph, u, v)

    return graph


def chain(graph, fs):
    for f in fs:
        graph = f(graph)
    return graph

if __name__ == "__main__":
    bnet = BayesNet('bnet')

    out = triangulate(moralize(bnet.graph))
