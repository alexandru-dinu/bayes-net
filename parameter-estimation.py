from utils import *

import numpy as np
from collections import namedtuple, defaultdict
from itertools import product
from termcolor import colored
from pprint import pprint


Query = namedtuple('Query', ['xs', 'given'])


class Samples:
    def __init__(self, filename):
        lines = [l.strip() for l in open(filename).readlines()]

        self.ordered_nodes = lines[0].split(" ")
        self.samples = {v: [] for v in self.ordered_nodes}

        for s in lines[1:]:
            vs = list(map(int, s.split(" ")))
            for i, v in enumerate(self.ordered_nodes):
                self.samples[v].append(vs[i])

        for v in self.ordered_nodes:
            self.samples[v] = np.array(self.samples[v])

        self.num_samples = len(lines[1:])


    def get_nth_sample(self, n):
        return np.array([self.samples[x][n] for x in self.ordered_nodes])


    def count(self, vs):
        """
        vs = [(var, value)]
        """

        common = set()

        x = vs[0]
        for i, v in enumerate(range(self.num_samples)):
            if self.samples[x[0]][i] == x[1]:
                common.add(i)

        for var, value in vs[1:]:
            a = set()
            for i in common:
                if self.samples[var][i] == value:
                    a.add(i)
            common &= a

        return len(common)

        # vs = dict(vs)

        # n = len(self.ordered_nodes)

        # _d = {x: 0 for x in self.ordered_nodes}
        # _d.update(vs)

        # _vars = dict(zip(self.ordered_nodes, range(len(self.ordered_nodes)-1, -1, -1)))

        # M = 2 ** n

        # p1 = 0
        # for i, v in enumerate(self.ordered_nodes):
        #     p1 |= (_d[v] << (n - 1 - i))

        # _d = {x: 0 for x in self.ordered_nodes}
        # for v in vs:
        #     _d[v] = 1

        # p2 = 0
        # for i, v in enumerate(self.ordered_nodes):
        #     p2 |= (_d[v] << (n - 1 - i))

        # total = 0
        # for i in range(self.num_samples):
        #     s = self.get_nth_sample(i)
        #     s = s.dot(1 << np.arange(s.size)[::-1])
        #     total += (((M + ~(s ^ p1)) & p2) >= p2)
        # return total


class BayesNet:
    def __init__(self, filename):
        self.graph = {}
        self.queries = []
        self.probabilities = {}
        self.samples = None
        self.num_samples = 0
        self.__ordered_nodes = None

        lines = [l.strip() for l in open(filename).readlines()]

        net_size, num_queries = list(map(int, lines[0].split(" ")))

        for i in range(net_size):
            line = lines[1 + i]
            n, parents, probs = [a.strip().split(" ") for a in line.split(";")]
            parents = [] if parents == [''] else parents
            self.graph[n[0]] = parents
            self.probabilities[n[0]] = list(map(float, probs))

        for i in range(num_queries):
            line = lines[1 + net_size + i]
            xs, gs = [x.strip() for x in line.split("|")]

            xs = [q.strip() for q in xs.split(" ")]
            xs = dict(map(lambda x : x.split("="), xs))
            xs = dict(zip(xs.keys(), map(int, xs.values())))

            gs = [q.strip() for q in gs.split(" ")]
            gs = [] if gs == [''] else gs
            gs = dict(map(lambda x : x.split("="), gs))
            gs = dict(zip(gs.keys(), map(int, gs.values())))

            self.queries.append(Query(xs, gs))


    def get_prob(self, node, given=None):
        """
        p(A | B, ~C) = get_prob("A", given="10")
        """
        if given is None:
            return self.probabilities[node]
        else:
            assert len(given) == len(self.graph[node])
            return self.probabilities[node][int(given, 2)]



def estimate_params(bnet, samples):
    params = defaultdict(lambda : [])

    for node, parents in bnet.graph.items():
        if parents == []:
            params[node] = samples.count([(node, 1)]) / samples.num_samples
        else:
            for prod in product(*[[0, 1]] * len(parents)):
                par = list(zip(parents, prod))
                count = round(
                    samples.count([(node, 1)] + par) / samples.count(par),
                    4
                )
                params[node].append(count)

    return params


if __name__ == "__main__":
    bnet = BayesNet("./input/pe_bnet")
    samples = Samples("./input/pe_samples")

    params = estimate_params(bnet, samples)

    for k, v in params.items():
        print(k, " - ", v)
