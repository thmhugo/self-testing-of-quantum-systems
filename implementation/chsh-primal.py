from math import sqrt
import numpy as np
from itertools import product
import collections
from gurobipy import *

# (x,y) domain and player names;
domain_xy = [0,1]
domain_ab = [-1,1]
players = ['a','b']

p = collections.defaultdict(int)
i = 0
for x,y in product(domain_xy, repeat=2):
    for a,b in product(domain_ab, repeat=2):
        p[a,b,x,y] = i
        i += 1

print(p)

# <ψ| A_i x B_j |ψ> = 1/sqrt(2) * ij
A1 = np.zeros([4,16])
B1 = np.zeros([4])
i = 0
for x,y in product(domain_xy, repeat=2):
    for a,b in product(domain_ab, repeat=2):
        A1[i,p[a,b,x,y]] = a*b
    B1[i] = (-1)**(x*y) / sqrt(2)
    i += 1

# <ψ| A_i x B_j |ψ> = 1
A2 = np.zeros([4,16])
B2 = np.zeros([4])
i = 0
for x,y in product(domain_xy, repeat=2):
    for a,b in product(domain_ab, repeat=2):
        A2[i,p[a,b,x,y]] = 1
    B2[i] = 1
    i += 1

# <ψ| A_i x 1_B |ψ> = 0
A3 = np.zeros([4,16])
B3 = np.zeros([4])
i=0
for x,y in product(domain_xy, repeat=2):
    for a,b in product(domain_ab, repeat=2):
        A3[i,p[a,b,x,y]] = a
    B3[i] = 0
    i += 1

# <ψ| 1_A x B_j |ψ> = 0
A4 = np.zeros([4,16])
B4 = np.zeros([4])
i=0
for x,y in product(domain_xy, repeat=2):
    for a,b in product(domain_ab, repeat=2):
        A4[i,p[a,b,x,y]] = b
    B4[i] = 0
    i += 1

# Constraint Coefficients
A = np.concatenate((A1,A2,A3,A4), axis=0)

# Values
B = np.concatenate((B1,B2,B3,B4), axis=None)

print(A)
print(B)

# Solve linear system
X = np.linalg.solve(A,B)

# P vector
P = X

print(X)

# linear program
m = Model()

def vec_d_lambda(l):
    dl = []
    for x,y in product(domain_xy, repeat=2):
        for a,b in product(domain_ab, repeat=2):
            dl.append(int(l[x] == a and l[y+2] == b))
    return dl

lambdas = []
for x,y in product(domain_xy, repeat=2):
    for a,b in product(domain_ab, repeat=2):
        lambdas.append((a,b,x,y))

mu_lambda = [m.addVar(name=f"mu_{i}",vtype="C") for i in range(len(lambdas))]

Q = m.addVar(name="Q",vtype="C")

D_l = [vec_d_lambda(l) for l in lambdas]

for i, p in enumerate(P):
    m.addConstr(Q*p + (1-Q) - quicksum(mu_lambda[j] * D_l[j][i] for j in range(len(lambdas))) <= 0)

m.addConstr(quicksum(mu_lambda[i] for i in range(len(lambdas))) == 1)

for i in range(len(lambdas)):
    m.addConstr(mu_lambda[i] >= 0)

m.update()

m.setObjective(Q, GRB.MINIMIZE)

m.optimize()
