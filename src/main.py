import itertools
from pprint import pprint


def vec_d_lambda(l):
    dl = []
    permutations = list(itertools.product([0, 1], repeat=2))

    for (x, y) in permutations:
        for (a, b) in permutations:
            dl.append(int(l[x] == a and l[y + 2] == b))

    return dl


def vec_p():
    return [
        int((a + b) % 2 == x * y)
        for a, b, x, y in list(itertools.product([0, 1], repeat=4))
    ]


lambdas = list(itertools.product([0, 1], repeat=4))
for l in lambdas:
    print(f"{l} -> {vec_d_lambda(l)}")

print(vec_p())
