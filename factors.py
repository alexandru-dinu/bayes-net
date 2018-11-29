from collections import namedtuple


Factor = namedtuple("Factor", ["vars", "values"])


def print_factor(phi, indent="\t"):
    line = " | ".join(phi.vars + ["ϕ(" + ",".join(phi.vars) + ")"])
    sep = "".join(["+" if c == "|" else "-" for c in list(line)])
    print(indent + sep)
    print(indent + line)
    print(indent + sep)
    for values, p in phi.values.items():
        print(indent + " | ".join([str(v) for v in values] + [str(p)]))
    print(indent + sep)


def factors_op(phi1, phi2, op):
    res_vars = sorted(list(set(phi1.vars).union(set(phi2.vars))))
    res_values = {}

    unique2 = list(set(phi2.vars).difference(set(phi1.vars)))
    common = list(set(phi1.vars).intersection(set(phi2.vars)))

    for (k1, p1) in phi1.values.items():
        for (k2, p2) in phi2.values.items():
            ok = True

            for var in common:
                # if the value for the common var differs, skip current pair
                if k1[phi1.vars.index(var)] != k2[phi2.vars.index(var)]:
                    ok = False
                    break

            if not ok:
                continue

            # all from k1 and common from k2
            ls = list(k1) + [k2[phi2.vars.index(var)] for var in unique2]
            res_values[tuple(ls)] = round(op(p1, p2), 3)

    return Factor(vars=res_vars, values=res_values)


def sum_out(var, phi):
    assert isinstance(phi, Factor) and var in phi.vars

    res_vars = [v for v in phi.vars if v != var]

    res_sum = {}

    for (k1, p1) in phi.values.items():
        for (k2, p2) in phi.values.items():
            # same value of var, skip
            if k1[phi.vars.index(var)] == k2[phi.vars.index(var)]:
                continue

            ok = True
            for v in res_vars:
                if k1[phi.vars.index(v)] != k2[phi.vars.index(v)]:
                    ok = False
                    break

            if not ok:
                continue

            ls = list(k1)
            ls.pop(phi.vars.index(var))

            res_sum[tuple(ls)] = p1 + p2


    return Factor(vars=res_vars, values=res_sum)
