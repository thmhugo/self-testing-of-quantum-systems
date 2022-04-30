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

N = (delta*m)**2

indexes_p = collections.defaultdict(int)  # Stores the index of each P(a,b,x,y)

i=0
for a, b in product(domain_ab, repeat=2):
    for x, y in product(domain_xy, repeat=2):
        indexes_p[a, b, x, y] = i
        i += 1

print("Basis P\n")
print(indexes_p)

def vec_d_lambda(l: int):
    """Generates the D_lambda vector associated to a lambda.

    Args:
        l (list[int]): a behavior lambda

    Returns:
        list[int]: the vector D_lambda
    """
    dl = []
    for a, b in list(product(domain_ab, repeat=2)):
        for x, y in list(product(domain_xy, repeat=2)):
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
    return [1 / 4.0] * N


p = ls_quantum_p()

R = np.ones(len(p))
for i in range(len(p)) :
    R[i] = 1/4.

# Create a new model
m = gp.Model()
# m.Params.LogToConsole = 0  # Less verbose Guroby output.

Y_p = [m.addVar(name=f"y_p_{i}", vtype="C") for i in range(N)]
Y_m = [m.addVar(name=f"y_m_{i}", vtype="C") for i in range(N)]

# Create variables

gamma_p = m.addVar(name="gamma_p", vtype="C")
gamma_m = m.addVar(name="gamma_m", vtype="C")

omega = m.addVar(name="omega", vtype="C")



m.update()


# Set objective function
m.setObjective(-gurobi_dot(p, Y_p)+gurobi_dot(p, Y_m)  + gamma_p - gamma_m -omega    , gp.GRB.MAXIMIZE)


# Add constraints

for l in lambdas:
    m.addConstr(gamma_p - gamma_m - gurobi_dot(Y_p, vec_d_lambda(l)) + gurobi_dot(Y_m, vec_d_lambda(l))   <= 0)

m.addConstr(gp.quicksum((R[i]-p[i])*(Y_p[i]-Y_m[i]) for i in range(len(p))) -omega <= 1)


m.update()
# Solve it!
m.optimize()
#m.display()


print(f"Optimal objective value S = {m.objVal}")
print(f"Solution values:     \n")
print(f"                        Y_p = {[Y_p[i].X for i in range(N)]}")
print(f"                        Y_m = {[Y_m[i].X for i in range(N)]}")
print(f"                        gamma_p = {gamma_p.X }")
print(f"                        gamma_m = {gamma_m.X }")
print(f"                        omega = {omega.X }")

print(f"               (recall) P = {p}")


print("dot R Y : ")
L =  [(Y_p[i].X - Y_m[i].X) for i in range(N)]
print(gurobi_dot(L,R))
print("dot P Y : ")
print(gurobi_dot(p,L))

print("dot Y Y : ")
print(gurobi_dot(L,L))

# evaluated_gurobi_dot = lambda a, b: sum(a[i].X * b[i] for i in range(len(b)))
#
# print(f"Inequality : s • p = {evaluated_gurobi_dot(Y, p)}" )
