from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

flatten = lambda l: [item for sublist in l for item in sublist]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_paths_util(graph, u, dest, visited, path, paths):
    visited[u] = True
    path.append(u)

    if u == dest:
        paths += [path[:]]
    else:
        for i in graph[u]:
            if not visited[i]:
                get_paths_util(graph, i, dest, visited, path, paths)

    path.pop()
    visited[u] = False


def get_paths(graph, src, dest):
    visited = {x: False for x in graph.keys()}
    paths = []

    get_paths_util(graph, src, dest, visited, [], paths)

    return paths


def is_edge(graph, u, v, no_dir=False):
    if no_dir:
        return (v in graph.get(u, [])) or (u in graph.get(v, []))
    else:
        return v in graph.get(u, [])


def is_closed(g, path, zs):
    """
    Check if path is closed w.r.t. z in zs
    """
    tr = get_transposed_graph(g)

    closed = []

    for i in range(len(path) - 2):
        x1, x2, x3 = path[i], path[i + 1], path[i + 2]

        # causal trail
        if is_edge(tr, x1, x2) and is_edge(tr, x2, x3):
            if x2 in zs:
                closed.append(True)

        # evidential trail
        if is_edge(tr, x3, x2) and is_edge(tr, x2, x1):
            if x2 in zs:
                closed.append(True)

        # common cause
        if is_edge(tr, x2, x1) and is_edge(tr, x2, x3):
            if x2 in zs:
                closed.append(True)

        # common effect
        if is_edge(tr, x1, x2) and is_edge(tr, x3, x2):
            if x2 in zs:
                closed.append(False)
            else:
                closed.append(True)

    return False if len(closed) == 0 else any(closed)


def get_edges(graph):
    edges = []
    for n, ps in graph.items():
        edges.extend([(p, n) for p in ps])
    return edges


def get_vertices(graph):
    return list(graph.keys())


def add_undir_edge(graph, u, v):
    graph[u].append(v)
    graph[v].append(u)
    return graph


def add_weighted_undir_edge(graph, u, v, w):
    graph[u][v] = w
    graph[v][u] = w
    return graph


def strip_weights(graph):
    out = {k: v.keys() for k, v in graph.items()}
    return out


def get_sorted_edges(graph, reverse=False):
    edges = [(n1, n2, w) for n1, ns in graph.items() for n2, w in ns.items()]
    return sorted(edges, key=itemgetter(2), reverse=reverse)


def show_graph(graph):
    G = nx.DiGraph()
    G.add_edges_from(get_edges(graph))
    # pos = nx.kamada_kawai_layout(G)
    pos = nx.spectral_layout(G)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, edge_color='k')
    plt.show()


def get_undirected_graph(graph):
    undir_graph = {}
    for k, v in graph.items():
        rest = [_k for _k, _v in graph.items() if k in _v]
        undir_graph[k] = v + rest
    return undir_graph


def get_transposed_graph(graph):
    tr = {}
    for k in graph.keys():
        tr[k] = [x for x, y in graph.items() if k in y]
    return tr
