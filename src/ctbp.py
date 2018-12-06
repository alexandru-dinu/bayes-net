from utils import *
from kruskal import kruskal

from copy import copy
from collections import defaultdict
from itertools import combinations
from operator import itemgetter
from pprint import pprint
import sys


class BayesNet:
    def __init__(self, filename):
        self.graph = {}
        self.probabilities = {}

        lines = [l.strip() for l in open(filename).readlines()]

        for line in lines:
            n, parents, probs = [a.strip().split(" ") for a in line.split(";")]
            parents = [] if parents == [''] else parents
            self.graph[n[0]] = parents
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


def BronKerbosch(graph, r, p, x, out):
    """
    finds the maximal cliques that include
    all of the vertices in R, some of the vertices in P, and none of the vertices in X
    """
    if len(p) == 0 and len(x) == 0:
        name = "".join(r)
        out[name] = r
    else:
        while p:
            v = p[0]

            BronKerbosch(graph,
                r=list(set(r + [v])),
                p=list(set(p).intersection(graph[v])),
                x=list(set(x).intersection(graph[v])),
                out=out
            )

            p.remove(v)
            x += [v]


def construct_weighted_clique_graph(graph):
    cliques = {}

    # find maximal cliques
    BronKerbosch(graph, r=[], p=list(graph.keys()), x=[], out=cliques)

    print(">>> [cliques]")
    pprint(cliques)

    clique_tree = {x: {} for x in cliques.keys()}
    pairs = combinations(cliques.keys(), 2)

    # add weighted edges between cliques
    for n1, n2 in pairs:
        c1, c2 = cliques[n1], cliques[n2]

        common = list(set(c1).intersection(set(c2)))
        size = len(common)

        if size == 0:
            continue

        add_weighted_undir_edge(clique_tree, n1, n2, size)

    # print(">>> [clique tree]")
    # show_graph(strip_weights(clique_tree))

    return clique_tree


def compute_max_spanning_tree(graph):
    mst = kruskal(graph)

    print(">>> [mst]")
    pprint(dict(mst))
    # show_graph(mst)

    return mst


def set_initial_factors(probs, clique_tree):

    for clique in clique_tree.keys():
        clique_nodes = list(map(lambda x : x, clique))



def belief_propagation(graph, query):
    pass


def chain(graph, *fs):
    g = copy(graph)
    for f in fs:
        g = f(g)
    return g


if __name__ == "__main__":
    bnet = BayesNet(sys.argv[1])

    print(">>> [bayes-net]")
    pprint(bnet.graph)

    out = chain(bnet.graph,
        moralize,
        triangulate,
        construct_weighted_clique_graph,
        compute_max_spanning_tree
    )

    set_initial_factors(bnet.probabilities, out)
