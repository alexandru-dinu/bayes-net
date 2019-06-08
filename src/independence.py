import pprint
import re
import sys

from collections import namedtuple
from termcolor import colored
import matplotlib.pyplot as plt
import networkx as nx

from utils import *


Query = namedtuple('Query', ['X', 'Y', 'Z'])


class Problem:
    def __init__(self, filename):
        self.graph = {}
        self.queries = []

        lines = [l.strip() for l in open(filename).readlines()]
        v, q = list(map(int, lines[0].split(" ")))

        # parse nodes
        for line in lines[1:v+1]:
            node_label, *parents_labels = line.split(" ")
            self.graph[node_label] = parents_labels

        # parse queries
        for line in lines[v+1:]:
            X, Y, *Z = [x.strip().split(" ") for x in re.findall("[A-Z\s]+", line)]
            self.queries.append(Query(X, Y, flatten(Z)))

    def show_queries(self):
        print("Queries:")
        for q in self.queries:
            print(q)
# -- problem


def solve(g, query):
    print(query)

    undir = get_undirected_graph(g)
    all_paths = flatten([get_paths(undir, x, y) for x in query.X for y in query.Y])

    res = []
    for path in all_paths:
        r = is_closed(g, path, query.Z)
        res.append(r)
        print(f"\t{'-'.join(path)}: closed = {r}")

    print(colored(f"Independent: {all(res)}", color='green' if all(res) else 'red'))


if __name__ == "__main__":
    problem = Problem(filename=sys.argv[1])

    for query in problem.queries:
        solve(problem.graph, query)
        print("\n")
