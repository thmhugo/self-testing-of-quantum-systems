from math import sqrt
from itertools import product
from random import uniform
from timeit import repeat
import collections
import gurobipy as gp


domain_xy = [0, 1]
domain_ab = [-1, 1]
delta = 2
m = 2

N = delta ** (2 * m)

indexes_p = collections.defaultdict(int)

# Stores the index of each P(a,b,x,y)
i = 0
for x, y in product(domain_xy, repeat=2):
    for a, b in product(domain_ab, repeat=2):
        indexes_p[a, b, x, y] = i
        i += 1


def vec_d_lambda(l: int):
    """Generates the D_lambda vector associated to a lambda.

    Args:
        l (list[int]): a behavior lambda

    Returns:
        list[int]: the vector D_lambda
    """
    dl = []
    for x, y in list(product([0, 1], repeat=2)):
        for a, b in list(product([-1, 1], repeat=2)):
            dl.append(int(l[x] == a and l[y + 2] == b))
    return dl


# lambdas = list(product([0, 1], repeat=4))
lambdas = []
for a0, a1 in list(product([-1, 1], repeat=2)):
    for b0, b1 in list(product([-1, 1], repeat=2)):
        lambdas.append((a0, a1, b0, b1))


def dot(A, B):
    """Returns the dot product A•B."""
    return gp.quicksum(A[i] * B[i] for i in range(N))


def prob(a, b, x, y):
    """Returns the winning probability P(a,b|x,y)."""
    if a == -1:
        a = 0
    if b == -1:
        b = 0
    return int((a + b) % 2 == x * y) * 0.5


def ns_p():
    """No signalling probability distribution"""
    print("NS\n")
    _p = []
    for x, y in product([0, 1], repeat=2):
        for a, b in product([-1, 1], repeat=2):
            _p.append(prob(a, b, x, y))
    return _p


def quantum_p():
    """Quantum probability distribution"""
    print("QUANTUM\n")
    minus = (sqrt(2) - 1) / (4 * sqrt(2))  # sin^2(pi/8)
    plus = (1 + sqrt(2)) / (4 * sqrt(2))  # cos^2(pi/8)
    return [  # (a, b, x, y)
        plus,  #  (-1, -1, 0, 0)
        plus,  #  (-1, -1, 0, 1)
        plus,  #  (-1, -1, 1, 0)
        minus,  # (-1, -1, 1, 1)
        minus,  # (-1,  1, 0, 0)
        minus,  # (-1,  1, 0, 1)
        minus,  # (-1,  1, 1, 0)
        plus,  #  (-1,  1, 1, 1)
        minus,  #  ( 1, -1, 0, 0)
        plus,  #  ( 1, -1, 0, 1)
        minus,  # ( 1, -1, 1, 0)
        minus,  # ( 1, -1, 1, 1)
        plus,  #  ( 1,  1, 0, 0)
        plus,  #  ( 1,  1, 0, 1)
        plus,  #  ( 1,  1, 1, 0)
        minus,  # ( 1,  1, 1, 1)
    ]


def uniform_p():
    """Uniform probability distribution"""
    print("UNIFORM\n")
    return [0.25 for _ in range(16)]


# Define the probability distribution we want to test
p = quantum_p()

## Just prints out the probabilities
# for x, y in product([0, 1], repeat=2):
#     for a, b in product([-1, 1], repeat=2):
#         print(f"P({a},{b}|{x},{y}) = {p[indexes_p[(a,b,x,y)]]} ")


# Create a new model
m = gp.Model()
# m.Params.LogToConsole = 0  # Less verbose Guroby output.


# Create variables
S = [m.addVar(name=f"s_{i}", vtype="C") for i in range(N)]
S_l = m.addVar(name="S_l", vtype="C")
m.update()


# Set objective function
m.setObjective(dot(S, p) - S_l, gp.GRB.MAXIMIZE)


# Add constraints
for l in lambdas:
    m.addConstr(dot(S, vec_d_lambda(l)) - S_l <= 0)

m.addConstr(dot(S, p) - S_l <= 1)


# Solve it!
m.optimize()


print(f"Optimal objective value S = {m.objVal}")
print(f"Solution values:      S_l = {S_l.X}")
print(f"                        S = {[S[i].X for i in range(N)]}")
print(f"               (recall) P = {p}")
dot_p = lambda A, B: gp.quicksum(A[i].X * B[i] for i in range(N))

print(f"Inequality : s • p = {dot_p(S, p).getConstant()} = S_l + 1 > S_l")

# m.display()
