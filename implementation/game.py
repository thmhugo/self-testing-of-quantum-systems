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


def uniform_noise(game):
    """_summary_

    Args:
        game (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [0.25] * game.N


def gurobi_dot(A, B):
    """Returns the dot product A•B.

    Args:
        A (_type_): _description_
        B (_type_): _description_

    Returns:
        _type_: _description_
    """
    return gp.quicksum(A[i] * B[i] for i in range(max(len(A), len(B))))


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
        for a0, a1 in list(product(domain_ab, repeat=2)):
            for b0, b1 in list(product(domain_ab, repeat=2)):
                self.lambdas.append((a0, a1, b0, b1))

        self.indexes_p = defaultdict(int)  # Stores the index of each P(a,b,x,y)
        i = 0
        for a, b in product(self.domain_ab, repeat=2):
            for x, y in product(self.domain_xy, repeat=2):
                self.indexes_p[a, b, x, y] = i
                i += 1

        self.d_lambda = np.zeros((len(self.lambdas), len(self.lambdas)))
        i = 0
        for l in self.lambdas:
            self.d_lambda[i] = np.array(self.vec_d_lambda(l))
            i += 1

        self.delta = len(self.domain_ab)
        self.m = len(self.domain_xy)
        self.N = (self.delta * self.m) ** 2

    def vec_d_lambda(self, l: int):
        """Generates the D_lambda vector associated to a lambda.

        Args:
            l (list[int]): a behavior lambda
            set (string): define wether

        Returns:
            list[int]: the vector D_lambda
        """
        dl = []

        for x, y in list(product(self.domain_xy, repeat=2)):
            for a, b in list(product(self.domain_ab, repeat=2)):
                dl.append(int(l[x] == a and l[y + self.offset] == b))
        return dl
