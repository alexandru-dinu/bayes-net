# bayes-net

- `independence.py`: check independence assumptions; i.e. (X ⊥ Y | Z)
- `ctbp.py`: clique trees [and **TODO: belief propagation**]
- `parameter-estimation.py`: estimate the parameters of a BN using two methods:
frequentist approach and parameter learning (using KL divergence)

## Input format

Currently, there are 4 input files:
1. [`ctbp_bnet`](https://github.com/alexandru-dinu/bayes-net/blob/master/input/ctbp_bnet)
    - for **c**lique **t**rees and **b**elief **p**ropagation
    - it's basically just the structure of the graph, formatted as `node ; parents ; (conditional) probability` (see [1])
2. [`pe_bnet`](https://github.com/alexandru-dinu/bayes-net/blob/master/input/pe_bnet)
    - bayesian network used for parameter estimation; the network follows the same format as 1.
    - queries to test belief propagation and their answers (this part may be moved to its own file)
3. [`pe_samples`](https://github.com/alexandru-dinu/bayes-net/blob/master/input/pe_samples)
    - 10k samples needed for **p**arameter **e**stimation
4. [`ind_queries`](https://github.com/alexandru-dinu/bayes-net/blob/master/input/ind_queries)
    - queries for conditional independence
    - test if a set of vars X is independent of a set of vars Y, given observed vars Z, i.e. `(X ⊥ Y | Z)` (see [2])

---

[1] For example,
- `A ; ; 0.8` - `A` is a node without any parents, with `p(A=1) = 0.8`
- `B ; A ; 0.8 0.1` - `B` is a node with `parents(B) = [A]` and, in order,
    - `p(B=1|A=0) = 0.8`
    - `p(B=1|A=1) = 0.1`
- `D ; B C ; 0.1 0.5 0.15 0.9` - `D` is a node with `parents(D) = [B,C]` and, in order,
    - `p(D=1|B=0,C=0) = 0.1`
    - `p(D=1|B=0,C=1) = 0.5`
    - `p(D=1|B=1,C=0) = 0.15`
    - `p(D=1|B=1,C=1) = 0.9`

and so on.

[2] Format:
- two positive numbers on the first line: N (number of nodes), M (number of queries)
- N lines with the name of each variable followed by the names of its parents
- M lines containing queries expressed as a sequence of names from X, the `;` symbol, the names from Y, the symbol `|`, and then the names of all observed variables from Z
- M lines containing either `true` or `false` corresponding to the correct answers of the M independence queries
