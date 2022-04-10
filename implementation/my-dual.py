from math import sqrt
from itertools import product
import collections
import gurobipy as gp
import numpy as np

np.set_printoptions(threshold=np.inf)


domain_xy = [0, 1, 2]
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


def vec_d_lambda(l: int):
    """Generates the D_lambda vector associated to a lambda.

    Args:
        l (list[int]): a behavior lambda

    Returns:
        list[int]: the vector D_lambda
    """
    dl = []
    for x, y in list(product(domain_xy, repeat=2)):
        for a, b in list(product(domain_ab, repeat=2)):
            dl.append(int(l[x] == a and l[y + 3] == b))
    return dl


lambdas = []
for a0, a1, a2 in list(product([-1, 1], repeat=3)):
    for b0, b1, b2 in list(product([-1, 1], repeat=3)):
        lambdas.append((a0, a1, a2, b0, b1, b2))


def gurobi_dot(A, B):
    """Returns the dot product A•B."""
    return gp.quicksum(A[i] * B[i] for i in range(N))


def ls_quantum_p():
    """
    Returns the probability distribution according to the Mayers-Yao
    correlations.
    """
    print("QUANTUM\n")
    # Constraint Coefficients
    A = np.zeros([N, N])
    # Left Side
    B = np.zeros([N])
    i = 0
    print(f"{N = }")

    for z in domain_xy:
        for a, b in product(domain_ab, repeat=2):
            A[i, indexes_p[a, b, z, z]] = a * b
        B[i] = 1
        i += 1

    for x in [0, 1]:
        y = 1 - x
        for a, b in product(domain_ab, repeat=2):
            A[i, indexes_p[a, b, x, y]] = a * b
        B[i] = 0
        i += 1

    for x, y in [(0, 2), (1, 2), (2, 0), (2, 1)]:
        for a, b in product(domain_ab, repeat=2):
            A[i, indexes_p[a, b, x, y]] = a * b
        B[i] = 1 / sqrt(2)
        i += 1

    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A[i, indexes_p[a, b, x, y]] = 1
        B[i] = 1
        i += 1

    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A[i, indexes_p[a, b, x, y]] = a
        B[i] = 0
        i += 1

    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A[i, indexes_p[a, b, x, y]] = b
        B[i] = 0
        i += 1

    # Solve linear system Ax = B
    return list(np.linalg.solve(A, B))


def uniform_p():
    print("UNIFORM\n")
    return [4 / 36.0] * N


p = ls_quantum_p()


# Create a new model
m = gp.Model()
m.Params.LogToConsole = 0  # Less verbose Guroby output.


# Create variables
S = [m.addVar(name=f"s_{i}", vtype="C") for i in range(N)]
S_l = m.addVar(name="S_l", vtype="C")
m.update()


# Set objective function
m.setObjective(gurobi_dot(S, p) - S_l, gp.GRB.MAXIMIZE)


# Add constraints
for l in lambdas:
    m.addConstr(gurobi_dot(S, vec_d_lambda(l)) - S_l <= 0)

m.addConstr(gurobi_dot(S, p) - S_l <= 1)


# Solve it!
m.optimize()


print(f"Optimal objective value S = {m.objVal}")
print(f"Solution values:      S_l = {S_l.X}")
print(f"                        s = {[S[i].X for i in range(N)]}")
print(f"               (recall) P = {p}")

evaluated_gurobi_dot = lambda a, b: sum(a[i].X * b[i] for i in range(len(b)))

print(f"Inequality : s • p = {evaluated_gurobi_dot(S, p)} = S_l + 1 > S_l")
