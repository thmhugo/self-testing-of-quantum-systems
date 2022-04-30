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
for a, b  in product(domain_ab, repeat=2):
    for x, y in product(domain_xy, repeat=2):
        indexes_p[a, b, x, y] = i
        i += 1
print(indexes_p)


def vec_d_lambda(l: int):
    """Generates the D_lambda vector associated to a lambda.

    Args:
        l (list[int]): a behavior lambda

    Returns:
        list[int]: the vector D_lambda
    """
    dl = []
    for a, b in list(product([-1, 1], repeat=2)):
        for x, y in list(product([0, 1], repeat=2)):
            dl.append(int(l[x] == a and l[y + 2] == b))
    return dl


lambdas = []  # stores every deterministric inputs
for a0, a1 in list(product([-1, 1], repeat=2)):
    for b0, b1 in list(product([-1, 1], repeat=2)):
        lambdas.append((a0, a1, b0, b1))

#D_l is a matrix with each row corresponding to a deterministic behavior lambda
D_l = np.zeros((len(lambdas),len(lambdas)))
i=0
for l in lambdas :
    D_l[i] = np.array(vec_d_lambda(l))
    i+=1

# M is the transpose of D_l , one column = one deterministic behavior d_lambda
M = np.column_stack(D_l)


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


def quantum_p():
    """Quantum probability distribution, calculated by hand."""
    print("QUANTUM\n")
    minus = (sqrt(2) - 1) / (4 * sqrt(2))  # sin^2(pi/8)
    plus = (1 + sqrt(2)) / (4 * sqrt(2))  # cos^2(pi/8)
    return [  #    (a, b, x, y)
        plus,  #   (-1, -1, 0, 0) 0
        plus,  #   (-1, -1, 0, 1) 1
        plus,  #   (-1, -1, 1, 0) 2
        minus,  #  (-1, -1, 1, 1) 3
        minus,  #  (-1,  1, 0, 0) 4
        minus,  #  (-1,  1, 0, 1) 5
        minus,  #  (-1,  1, 1, 0) 6
        plus,  #   (-1,  1, 1, 1) 7
        minus,  #  ( 1, -1, 0, 0) 8
        minus,  #  ( 1, -1, 0, 1) 9
        minus,  #  ( 1, -1, 1, 0) 10
        minus,  #  ( 1, -1, 1, 1) 11
        plus,  #   ( 1,  1, 0, 0) 12
        plus,  #   ( 1,  1, 0, 1) 13
        plus,  #   ( 1,  1, 1, 0) 14
        minus,  #  ( 1,  1, 1, 1) 15
    ]


def ls_quantum_p():
    """Quantum probability distribution, but computed by solving the system of equations."""
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
# p = uniform_p()

print("P\n")
print(p)

# Create a new model
m = gp.Model()
m.Params.LogToConsole = 0

R = np.zeros(len(p))
for i in range(len(R)) :
    R[i] = 1/4.


# Create variables
Y = [m.addVar(name=f"y_{i}", vtype="C") for i in range(N)]
gamma_p = m.addVar(name="gamma_p", vtype="C")
gamma_m = m.addVar(name="gamma_m", vtype="C")

m.update()


# Set objective function
m.setObjective(gurobi_dot(p, Y) + gamma_p - gamma_m    , gp.GRB.MAXIMIZE)


# Add constraints

i=0
for l in lambdas:
    m.addConstr(gamma_p - gamma_m  + gurobi_dot(Y, vec_d_lambda(l)) <= 0)

m.addConstr(gp.quicksum((-R[i]+p[i])*(Y[i]) for i in range(len(p)))   <= 1)

m.update()
# Solve it!
m.optimize()


print(f"Optimal objective value S = {m.objVal}")
print(f"Solution values:     \n")
print(f"                        Y = {[Y[i].X for i in range(N)]}")
print(f"                        gamma_p = {gamma_p.X }")
print(f"                        gamma_m = {gamma_m.X }")
print(f"               (recall) P = {p}")
#


print("dot R Y : ")
L= [(Y[i].X) for i in range(N)]
print(gurobi_dot(L,R))
print("dot P Y : ")
print(gurobi_dot(p,L))

print("dot Y Y : ")
print(gurobi_dot(L,L))