from utils import *

from collections import namedtuple


Query = namedtuple('Query', ['X', 'Given'])


class BayesNet:
    def __init__(self, filename):
        self.graph = {}
        self.queries = []
        self.probabilities = {}
        self.samples = None
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


    def read_samples(self, filename):
        lines = [l.strip() for l in open(filename).readlines()]

        self.__ordered_nodes = lines[0].split(" ")
        self.samples = {v: [] for v in self.__ordered_nodes}

        for s in lines[1:]:
            vs = list(map(int, s.split(" ")))
            for i, v in enumerate(self.samples.keys()):
                self.samples[v].append(vs[i])


    def get_nth_sample(self, n):
        return [self.samples[x][n] for x in self.__ordered_nodes]


    def get_prob(self, node, given=None):
        """
        p(A | B, ~C) = get_prob("A", given="10")
        """
        if given is None:
            return self.probabilities[node]
        else:
            assert len(given) == len(self.graph[node])
            return self.probabilities[node][int(given, 2)]



if __name__ == "__main__":
    bnet = BayesNet("./input/pe_bnet")
    bnet.read_samples("./input/pe_samples")
