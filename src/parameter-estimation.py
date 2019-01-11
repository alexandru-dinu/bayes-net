import sys
from collections import namedtuple, defaultdict
from functools import reduce
from itertools import product

from utils import *

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

    def get_indices_of(self, vs):
        # vs = [(var, value)]

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

        return np.array(list(common))

    def count(self, vs):
        # vs = [(var, value)]

        head, *tail = vs

        common = reduce(
            lambda acc, e: np.intersect1d(acc, np.where(self.samples[e[0]] == e[1])),
            tail,
            np.where(self.samples[head[0]] == head[1])[0]
        )

        return len(common)


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
            xs = dict(map(lambda x: x.split("="), xs))
            xs = dict(zip(xs.keys(), map(int, xs.values())))

            gs = [q.strip() for q in gs.split(" ")]
            gs = [] if gs == [''] else gs
            gs = dict(map(lambda x: x.split("="), gs))
            gs = dict(zip(gs.keys(), map(int, gs.values())))

            self.queries.append(Query(xs, gs))

    def get_prob(self, node, given=None):
        # p(A | B, ~C) = get_prob("A", given="10")

        if given is None:
            return self.probabilities[node]
        else:
            assert len(given) == len(self.graph[node])
            return self.probabilities[node][int(given, 2)]


class Param:
    def __init__(self, X, Y, val=0):
        self.X = X
        self.Y = Y
        self.val = val

    def __str__(self):
        return f"{self.X} | {self.Y}; val = {round(sigmoid(self.val), 4)}"


# Frequentist approach

def estimate_params(bnet, samples):
    params = defaultdict(lambda: [])

    for node, parents in bnet.graph.items():
        if not parents:
            params[node] = samples.count([(node, 1)]) / samples.num_samples
        else:
            for prod in product(*[[0, 1]] * len(parents)):
                par = list(zip(parents, prod))
                count = round(samples.count([(node, 1)] + par) / samples.count(par), 4)
                params[node].append(count)

    return params


# Learn parameters

def construct_params(bnet):
    params = []

    for x, parents in bnet.graph.items():
        if not parents:
            params.append(Param(X=x, Y={}))
        else:
            for prod in product(*[[0, 1]] * len(parents)):
                xs = dict(zip(parents, prod))
                params.append(Param(X=x, Y=xs))

    return params


def init_params(params):
    for param in params:
        param.val = np.random.uniform()
    return params


def learn_params(bnet, samples, num_iter=10, lr=0.001):
    params = construct_params(bnet)
    params = init_params(params)

    # learning loop
    for it in range(num_iter):
        for param in params:

            # no "givens"
            if param.Y == {}:
                subset = np.arange(samples.num_samples)
            # get samples subset for parent values
            else:
                subset = samples.get_indices_of(list(param.Y.items()))

            # perform update
            for i in subset:
                param.val += lr * (samples.samples[param.X][i] - sigmoid(param.val))

        print(f"[+] Iter {it+1}/{num_iter}")

    return params


def main(args):
    bnet = BayesNet(filename=args[0])
    samples = Samples(filename=args[1])

    # params = estimate_params(bnet, samples)
    params = learn_params(bnet, samples, num_iter=20, lr=1e-3)

    for p in params:
        print(p)


if __name__ == "__main__":
    main(sys.argv[1:])
