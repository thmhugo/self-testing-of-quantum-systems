from itertools import product
from collections import defaultdict
import numpy as np
import gurobipy as gp
from math import sqrt


def quantum_probability_distribution_chsh(game):
    """Computes the probability distribution described by the expectation
        values

    Args:
        game (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Constraint Coefficients
    A = np.zeros([16, 16])
    # Left Side
    B = np.zeros([16])
    i = 0

    # <ψ| A_x x B_y |ψ> = xy/sqrt(2)
    for x, y in product(game.domain_xy, repeat=2):
        for a, b in product(game.domain_ab, repeat=2):
            A[i, game.indexes_p[a, b, x, y]] = a * b
        B[i] = (-1) ** (x * y) / sqrt(2)
        i += 1

    # <ψ| A_x x B_y |ψ> = 1
    for x, y in product(game.domain_xy, repeat=2):
        for a, b in product(game.domain_ab, repeat=2):
            A[i, game.indexes_p[a, b, x, y]] = 1
        B[i] = 1
        i += 1

    # <ψ| A_x x 1_B |ψ> = 0
    for x, y in product(game.domain_xy, repeat=2):
        for a, b in product(game.domain_ab, repeat=2):
            A[i, game.indexes_p[a, b, x, y]] = a
        B[i] = 0
        i += 1

    # <ψ| 1_A x B_y |ψ> = 0
    for x, y in product(game.domain_xy, repeat=2):
        for a, b in product(game.domain_ab, repeat=2):
            A[i, game.indexes_p[a, b, x, y]] = b
        B[i] = 0
        i += 1

    # Solve linear system Ax = B
    return list(np.linalg.solve(A, B))


def no_signaling_probability_distribution_chsh(game):
    def prob(a, b, x, y):
        """Returns the winning probability P(a,b|x,y)."""
        if a == -1:
            a = 0
        if b == -1:
            b = 0
        return int((a + b) % 2 == x * y) * 0.5

    _p = []
    for x, y in product(game.domain_xy, repeat=2):
        for a, b in product(game.domain_ab, repeat=2):
            _p.append(prob(a, b, x, y))
    return _p


def quantum_probability_distribution_mayers_yao(game):
    """_summary_

    Args:
        game (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = (game.delta * game.m) ** 2

    # Constraint Coefficients
    A = np.zeros([n, n])
    # Left Side
    B = np.zeros([n])
    i = 0

    for z in game.domain_xy:
        for a, b in product(game.domain_ab, repeat=2):
            A[i, game.indexes_p[a, b, z, z]] = a * b
        B[i] = 1
        i += 1

    for x in [0, 1]:
        y = 1 - x
        for a, b in product(game.domain_ab, repeat=2):
            A[i, game.indexes_p[a, b, x, y]] = a * b
        B[i] = 0
        i += 1

    for x, y in [(0, 2), (1, 2), (2, 0), (2, 1)]:
        for a, b in product(game.domain_ab, repeat=2):
            A[i, game.indexes_p[a, b, x, y]] = a * b
        B[i] = 1 / sqrt(2)
        i += 1

    for x, y in product(game.domain_xy, repeat=2):
        for a, b in product(game.domain_ab, repeat=2):
            A[i, game.indexes_p[a, b, x, y]] = 1
        B[i] = 1
        i += 1

    for x, y in product(game.domain_xy, repeat=2):
        for a, b in product(game.domain_ab, repeat=2):
            A[i, game.indexes_p[a, b, x, y]] = a
        B[i] = 0
        i += 1

    for x, y in product(game.domain_xy, repeat=2):
        for a, b in product(game.domain_ab, repeat=2):
            A[i, game.indexes_p[a, b, x, y]] = b
        B[i] = 0
        i += 1

    # Solve linear system Ax = B
    return list(np.linalg.solve(A, B))


def uniform_noise(game):
    """_summary_

    Args:
        game (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [0.25] * (game.delta * game.m) ** 2


def gurobi_dot(A, B):
    """Returns the dot product A•B.

    Args:
        A (_type_): _description_
        B (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(A) != len(B):
        raise ValueError(
            f"A and B should have the same length. Here {len(A) = } != {len(B) = }"
        )
    return gp.quicksum(A[i] * B[i] for i in range(len(A)))


def evaluated_gurobi_dot(A, B):
    return gurobi_dot(A, B).getConstant()


def chsh_value(game, P):
    """_summary_

    Args:
        game (_type_): _description_
        P (_type_): _description_

    Returns:
        _type_: _description_
    """
    E = [0] * 4
    i = 0
    for x, y in product(game.domain_xy, repeat=2):
        s = 0
        for a, b in product(game.domain_ab, repeat=2):
            s += a * b * P[game.indexes_p[a, b, x, y]]
        E[i] = (-1) ** (x * y) * s
        i += 1
    return sum(e for e in E)


class Game:
    def __init__(self, domain_xy, domain_ab):
        """_summary_

        Args:
            domain_xy (_type_): _description_
            domain_ab (_type_): _description_
        """
        self.domain_xy = domain_xy
        self.domain_ab = domain_ab
        self.offset = len(domain_xy)
        self.model = gp.Model()
        self.model.Params.LogToConsole = 0

        self.lambdas = []

        for a in list(product(self.domain_ab, repeat=self.offset)):
            for b in list(product(self.domain_ab, repeat=self.offset)):
                self.lambdas.append((*a, *b))

        self.delta = len(self.domain_ab)
        self.m = len(self.domain_xy)
        self.N = self.delta ** (self.m * 2)

        self.indexes_p = defaultdict(int)  # Stores the index of each P(a,b,x,y)
        i = 0
        for x, y in product(self.domain_xy, repeat=2):
            for a, b in product(self.domain_ab, repeat=2):
                self.indexes_p[a, b, x, y] = i
                i += 1

        self.d_lambda = np.zeros((self.N, (self.delta * self.m) ** 2))
        i = 0
        for l in self.lambdas:
            self.d_lambda[i] = np.array(self.vec_d_lambda(l))
            i += 1

        # M is the transpose of d_lambda , one column = one deterministic
        # behavior d_lambda
        self.M = np.column_stack(self.d_lambda)

    def vec_d_lambda(self, l: int):
        """Generates the D_lambda vector associated to a lambda.

        Args:
            l (list[int]): a behavior lambda

        Returns:
            list[int]: the vector D_lambda
        """
        dl = []
        for x, y in list(product(self.domain_xy, repeat=2)):
            for a, b in list(product(self.domain_ab, repeat=2)):
                dl.append(int(l[x] == a and l[y + self.offset] == b))
        return dl
