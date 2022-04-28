from math import sqrt
from itertools import product
import collections
import gurobipy as gp
import numpy as np

np.set_printoptions(threshold=np.inf)


domain_x = [0, 1]
domain_y = [0, 1, 2]
domain_ab = [-1, 1]
delta = len(domain_ab)

N = (delta**2)*len(domain_x)*len(domain_y)

 # Stores the index of each P(a,b,x,y)
indexes_p = collections.defaultdict(int)
i = 0
for a, b in product(domain_ab, repeat=2):
    for x in domain_x:
        for y in domain_y:
            indexes_p[a, b, x, y] = i
            i += 1

print("\n ------  basis order\n")
print(indexes_p)
print("\n\n")

#-------------------------------------------------------------------------------
#       DETRMINISTIC BEHAVIOR
#-------------------------------------------------------------------------------

def vec_d_lambda(l: int):
    """Generates the D_lambda vector associated to a lambda.
    Args:
        l (list[int]): a behavior lambda
    Returns:
        list[int]: the vector D_lambda
    """
    dl = []
    for x in domain_x:
        for y in domain_y:
            for a, b in list(product(domain_ab, repeat=2)):
                dl.append(int(l[x] == a and l[y + 3] == b))
    return dl

# Lambdas are the possible outputs assignement (a0,a1,a2,b0,b1,b2), there are 64 possible lambdas

lambdas = []
for a0, a1, a2 in list(product([-1, 1], repeat=3)):
    for b0, b1, b2 in list(product([-1, 1], repeat=3)):
        lambdas.append((a0, a1, a2, b0, b1, b2))



def gurobi_dot(A, B):
    """Returns the dot product A•B."""
    return gp.quicksum(A[i] * B[i] for i in range(N))

#-------------------------------------------------------------------------------
#       QUANTUM CORR
#-------------------------------------------------------------------------------

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

    for z in domain_x:
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

    for x, y in [(0, 2), (1, 2)]:
        for a, b in product(domain_ab, repeat=2):
            A[i, indexes_p[a, b, x, y]] = a * b
        B[i] = 1 / sqrt(2)
        i += 1

    for x in domain_x:
        for y in domain_y:
            for a, b in product(domain_ab, repeat=2):
                A[i, indexes_p[a, b, x, y]] = 1
            B[i] = 1
            i += 1

    for x in domain_x:
        for y in domain_y:
            for a, b in product(domain_ab, repeat=2):
                A[i, indexes_p[a, b, x, y]] = a
            B[i] = 0
            i += 1

    for x in domain_x:
        for y in domain_y:
            for a, b in product(domain_ab, repeat=2):
                A[i, indexes_p[a, b, x, y]] = b
            B[i] = 0
            i += 1
    print("\n I  = ")
    print(i)
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
m.Params.LogToConsole = 0  # Less verbose Guroby output.

Y = [m.addVar(name=f"y_{i}", vtype="C") for i in range(N)]

# Create variables

gamma_p = m.addVar(name="gamma_p", vtype="C")
gamma_m = m.addVar(name="gamma_m", vtype="C")



m.update()


# Set objective function
m.setObjective(gurobi_dot(p, Y) + gamma_p - gamma_m    , gp.GRB.MAXIMIZE)


# Add constraints

for l in lambdas:
    m.addConstr(gamma_p - gamma_m + gurobi_dot(Y, vec_d_lambda(l)) <= 0)

m.addConstr(gp.quicksum((-R[i]+p[i])*(Y[i]) for i in range(len(p))) <= 1)


m.update()
# Solve it!
m.optimize()
#m.display()


print(f"Optimal objective value S = {m.objVal}")
print(f"Solution values:     \n")
print(f"                        Y = {[Y[i].X for i in range(N)]}")
print(f"                        gamma_p = {gamma_p.X }")
print(f"                        gamma_m = {gamma_m.X }")

print(f"               (recall) P = {p}")


print("dot R Y : ")
L= [(Y[i].X) for i in range(N)]
print(gurobi_dot(L,R))
print("dot P Y : ")
print(gurobi_dot(p,L))

print("dot Y Y : ")
print(gurobi_dot(L,L))

# evaluated_gurobi_dot = lambda a, b: sum(a[i].X * b[i] for i in range(len(b)))
#
# print(f"Inequality : s • p = {evaluated_gurobi_dot(Y, p)}" )
