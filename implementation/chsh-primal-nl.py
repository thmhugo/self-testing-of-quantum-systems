from math import sqrt
import numpy as np
from itertools import product
import collections
from gurobipy import *

# inputs (x,y) domain and outputs (a,b) domain
domain_xy = [0, 1]
domain_ab = [-1, 1]

# dictionnary giving the basis order, i.e p[(a,b,x,y)] = index of p(ab|xy) in the basis order chosen
p = collections.defaultdict(int)
i = 0
for x, y in product(domain_xy, repeat=2):
    for a, b in product(domain_ab, repeat=2):
        p[a, b, x, y] = i
        i += 1


def vec_d_lambda(l):
    dl = []
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            dl.append(int(l[x] == a and l[y + 2] == b))
    return dl


# Lambdas are the possible outputs assignement (a0,a1,b0,b1), there are 16 possible lambdas
lambdas = []
for a0, a1 in product(domain_ab, repeat=2):
    for b0, b1 in product(domain_ab, repeat=2):
        lambdas.append((a0, a1, b0, b1))

# D_l is a matrix with each row corresponding to a deterministic behavior lambda
D_l = [np.array(vec_d_lambda(l)) for l in lambdas]

# M is the transpose of D_l , one column = one deterministic behavior d_lambda
M = np.column_stack(D_l)


# define the model
m = Model()
m.Params.LogToConsole = 0
m.params.NonConvex = 2  # Needed since the model is non convex.

random = [0.25] * 16

# mu_lambda is a vector of the coeff of the linear combination of the vectors d_lambda
mu_lambda = [m.addVar(name=f"mu_{i}", vtype="C") for i in range(len(lambdas))]

P = [m.addVar(name=f"p_{i}", vtype="C") for i in range(16)]
# P_l : vector  of the convex combination of the deterministic points,
# i.e P_l = sum(mu_lambda * vec_d_lambda) where the sum is on the lambdas
P_l = np.dot(M, mu_lambda)


# add a variable Q (visibility)
Q = m.addVar(name="Q", vtype="C")


# update the model with the newly defined variables
m.update()

# Add the constraints
for i in range(len(P)):
    m.addConstr(((1 - Q) * P[i] + Q - P_l[i] == 0))

# Constraints on probabilities
m.addConstr(quicksum(mu_lambda[i] for i in range(len(lambdas))) >= 1)

for i in range(len(lambdas)):
    m.addConstr(mu_lambda[i] >= 0)
    m.addConstr(P[i] >= 0)

# Define a constraint for the chsh inequality
chsh = LinExpr()
for x, y in product(domain_xy, repeat=2):
    e = LinExpr()  # represents an expecation value <AxBy>
    for a, b in product(domain_ab, repeat=2):
        e += a * b * P[p[a, b, x, y]]
    # Constraints from (10) p.6 : Bell nonlocality
    m.addConstr(e <= 1)
    m.addConstr(e >= -1)
    chsh += (-1) ** (x * y) * e

for x, y in product(domain_xy, repeat=2):
    # Each marginal must sum up to one
    m.addConstr(
        quicksum(P[p[a, b, x, y]] for a, b in product(domain_ab, repeat=2)) == 1
    )
# If useless, removed by guroby
m.addConstr(quicksum(P[i] for i in range(len(lambdas))) == 4)

for x, y in product(domain_xy, repeat=2):
    m.addConstr(
        quicksum(P[p[a, b, x, y]] for a, b in product(domain_ab, repeat=2)) == 1
    )

for x, y in product(domain_xy, repeat=2):
    m.addConstr(
        quicksum(a * P[p[a, b, x, y]] for a, b in product(domain_ab, repeat=2)) == 0
    )

for x, y in product(domain_xy, repeat=2):
    m.addConstr(
        quicksum(b * P[p[a, b, x, y]] for a, b in product(domain_ab, repeat=2)) == 0
    )


m.addConstr(Q >= 0)
m.addConstr(Q <= 1)

m.addConstr(chsh >= 2 * sqrt(2))

m.setObjective(chsh, GRB.MINIMIZE)

m.update()
m.optimize()

print(f"Objective = {m.objVal}")
print(f"{Q.X = }")
print(f"Computed probability distribution : \n{np.array([P[i].X for i in range(16)])}")
print(f"CHSH value : {chsh.getValue()}")
