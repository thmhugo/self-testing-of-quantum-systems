from difflib import unified_diff
from math import sqrt
from itertools import product
import collections
from random import uniform
import gurobipy as gp
import numpy as np

domain_xy = [0, 1]
domain_ab = [-1, 1]
delta = len(domain_ab)
m = len(domain_xy)

N = (delta * m) ** 2

indexes_p = collections.defaultdict(int)  # Stores the index of each P(a,b,x,y)

i = 0
for a, b in product(domain_ab, repeat=2):
    for x, y in product(domain_xy, repeat=2):
        indexes_p[a, b, x, y] = i
        i += 1
# print(indexes_p)


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


lambdas = []  # stores every deterministric inputs
for a0, a1 in list(product([-1, 1], repeat=2)):
    for b0, b1 in list(product([-1, 1], repeat=2)):
        lambdas.append((a0, a1, b0, b1))


def gurobi_dot(A, B):
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


def ls_quantum_p():
    """
    Quantum probability distribution,
    computed by solving the system of equations.
    """
    print("QUANTUM LS\n")
    # Constraint Coefficients
    A = np.zeros([16, 16])
    # Left Side
    B = np.zeros([16])
    i = 0

    # <ψ| A_x x B_y |ψ> = xy/sqrt(2)
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A[i, indexes_p[a, b, x, y]] = a * b
        B[i] = (-1) ** (x * y) / sqrt(2)
        i += 1

    # <ψ| A_x x B_y |ψ> = 1
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A[i, indexes_p[a, b, x, y]] = 1
        B[i] = 1
        i += 1

    # <ψ| A_x x 1_B |ψ> = 0
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A[i, indexes_p[a, b, x, y]] = a
        B[i] = 0
        i += 1

    # <ψ| 1_A x B_y |ψ> = 0
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A[i, indexes_p[a, b, x, y]] = b
        B[i] = 0
        i += 1

    # Solve linear system Ax = B
    return list(np.linalg.solve(A, B))


def uniform_p():
    """Uniform probability distribution"""
    print("UNIFORM\n")
    return [0.25] * N


# Define the probability distribution we want to test
p = ls_quantum_p()
# p = ()

# Create a new model
m = gp.Model()
m.Params.LogToConsole = 0  # Less verbose Gurobi output.


# Create variables
S = [m.addVar(name=f"s_{i}", vtype="C") for i in range(N)]
phi = m.addVar(name="phi", vtype="C")
m.update()


# Set objective function
m.setObjective(phi, gp.GRB.MAXIMIZE)


# Add constraints
for l in lambdas:
    d = vec_d_lambda(l)
    m.addConstr(gp.quicksum(-S[i] * d[i] for i in range(16)) + phi <= 0)

# m.addConstr(gurobi_dot(S, p) >= 1)
m.addConstr(gp.quicksum(S[i] * p[i] for i in range(16)) <= 1)
m.update()

# m.addConstr(phi >= 0)

# Solve it!
m.optimize()

# m.display()

print(f"Optimal objective value S = {m.objVal}")
print(f"Solution values:      phi = {phi.X}")
print(f"                        s = \n{np.array([S[i].X for i in range(N)])}")
print(f"               (recall) P = \n{np.array(p)}")

evaluated_gurobi_dot = lambda a, b: sum(a[i].X * b[i] for i in range(len(b)))

print(f"s • p = {evaluated_gurobi_dot(S, p)} ")

primal = m.getAttr("Pi", m.getConstrs())
# print(sum(s.X for s in S))
print(f"{primal = }")

# for c in m.getConstrs():
#     print(f"The dual value of {c.constrName} : {c.pi}")

i = 0
for a, b in product(domain_ab, repeat=2):
    for x, y in product(domain_xy, repeat=2):
        if S[i].X != 0:
            print(f"{S[i].X} -- {p[indexes_p[a,b,x,y]]}")
        i += 1
