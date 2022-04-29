from math import sqrt
import numpy as np
from itertools import product
import collections
from gurobipy import *
import scipy.linalg as spla

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

print("\n ------  basis order\n")
print(p)
print("\n\n")

#-------------------------------------------------------------------------------
#       QUANTUM CORR
#-------------------------------------------------------------------------------

def quantum_corr(m):
        # """
        # Returns the probability distribution according to the CHSH
        # correlations.
        # """
    m = 1/m
    # <ψ| A_i x B_j |ψ> = 1/sqrt(2) * ij
    A1 = np.zeros([4, 16])
    B1 = np.zeros([4])
    i = 0
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A1[i, p[a, b, x, y]] = a * b * m
        B1[i] = (-1) ** (x * y)/sqrt(2)
        i += 1

    # <ψ| A_i x B_j |ψ> = 1
    A2 = np.zeros([4, 16])
    B2 = np.zeros([4])
    i = 0
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A2[i, p[a, b, x, y]] = 1
        B2[i] = 1
        i += 1

    # <ψ| A_i x 1_B |ψ> = 0
    A3 = np.zeros([4, 16])
    B3 = np.zeros([4])
    i = 0
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A3[i, p[a, b, x, y]] = a
        B3[i] = 0
        i += 1
    # <ψ| 1_A x B_j |ψ> = 0
    A4 = np.zeros([4, 16])
    B4 = np.zeros([4])
    i = 0
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A4[i, p[a, b, x, y]] = b
        B4[i] = 0
        i += 1

    # Constraint Coefficients
    A = np.concatenate((A1, A2, A3, A4), axis=0)

    # Values
    B = np.concatenate((B1, B2, B3, B4), axis=None)
    # # Solve linear system AP = B to have the vector of quantum correlations
    P = spla.solve(A, B)




    # print("\n ------  Quantum Correlations (CHSH)")
    # print(P) ;
    # print("\n\n")

    return P

#-------------------------------------------------------------------------------
#       LOCAL CORR
#-------------------------------------------------------------------------------

def local_corr():
    P = np.zeros(len(domain_ab)**2*len(domain_xy)**2)
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            index = p[(a,b,x,y)]
            P[index]= 1/4.

    return(P)

#-------------------------------------------------------------------------------
#       DETRMINISTIC BEHAVIOR
#-------------------------------------------------------------------------------

def vec_d_lambda(l):
    dl = []
    sum = 0
    i=0
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            dl.append(int(l[x] == a and l[y + 2] == b))
    return dl

# Lambdas are the possible outputs assignement (a0,a1,b0,b1), there are 16 possible lambdas
lambdas = []
for a0, a1 in product(domain_ab, repeat=2):
    for b0, b1 in product(domain_ab, repeat=2):
        lambdas.append((a0, a1, b0, b1))

#D_l is a matrix with each row corresponding to a deterministic behavior lambda
D_l = np.zeros((len(lambdas),len(lambdas)))
i=0
for l in lambdas :
    D_l[i] = np.array(vec_d_lambda(l))
    i+=1

# M is the transpose of D_l , one column = one deterministic behavior d_lambda
M = np.column_stack(D_l)


#-------------------------------------------------------------------------------
#       LINEAR PROGRAMMING WITH GUROBI SOLVER
#-------------------------------------------------------------------------------

a = 1
stop =0
step = 1e-4

P = quantum_corr(a)
print(P)

random = np.zeros(len(P))
for i in range(len(P)) :
    random[i] = 1/4.

while (stop==0 and a>step) :

    a -= step

    #define the model
    m = Model()
    m.Params.LogToConsole = 0

    P = quantum_corr(a)


    mu_lambda = [m.addVar(name=f"mu_{i}", vtype="C") for i in range(len(lambdas))]
    P_l = np.dot(M,mu_lambda)

    Q = m.addVar(name="Q", vtype="C")

    m.update()

    for i in range(len(P)):
        m.addConstr(((1-Q)*P[i]  + Q*random[i] <= P_l[i])  )

    m.addConstr(quicksum(mu_lambda[i] for i in range(len(lambdas))) == 1)

    for i in range(len(lambdas)):
        m.addConstr(mu_lambda[i] >= 0)

    # objective = Min Q
    m.setObjective(Q, GRB.MINIMIZE)

    m.update()

    m.optimize()
    #print(f"                        mu_lambda= {[mu_lambda[i].X for i in range(len(lambdas))]}")
    
    print(f"\n                        Q = {Q.X }")
    print(f"                          a = {a}")
    print(f"               (recall) P = {P}\n")

    delta = (m.objVal)**2


    if (m.objVal<1e-4):
        stop = 1
        print("STOP")

    m.remove(m.getVars())
    m.remove(m.getConstrs())
    m.reset(0)


print(f"Value of a = {a} ")
