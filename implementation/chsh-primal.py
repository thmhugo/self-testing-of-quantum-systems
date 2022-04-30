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

print("\n ------  basis order\n")
print(p)
print("\n\n")

#-------------------------------------------------------------------------------
#       QUANTUM CORR
#-------------------------------------------------------------------------------

def quantum_corr():
        # """
        # Returns the probability distribution according to the CHSH
        # correlations.
        # """
    # <ψ| A_i x B_j |ψ> = 1/sqrt(2) * ij
    A1 = np.zeros([4, 16])
    B1 = np.zeros([4])
    i = 0
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            A1[i, p[a, b, x, y]] = a * b
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
    print(A3)
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

    # Solve linear system AP = B to have the vector of quantum correlations
    P = np.linalg.solve(A, B)

    print("\n ------  Quantum Correlations (CHSH)")
    print(P) ;
    print("\n\n")

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

    print("\n ------  Local Correlations")
    print(P) ;
    print("\n\n")

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

#define the model
m = Model()

print("\n ----------------------------------\n Enter 0 for local correlations or 1 for quantum correlations \n ")
k = int(input())

if k==0 :
    P = local_corr()
if k==1 :
    P = quantum_corr()

random = np.zeros(len(P))
for i in range(len(P)) :
    random[i] = 1/4.

s# mu_lambda is a vector of the coeff of the linear combination of the vectors d_lambda
mu_lambda = [m.addVar(name=f"mu_{i}", vtype="C") for i in range(len(lambdas))]

# P_l : vector  of the convex combination of the deterministic points,
# i.e P_l = sum(mu_lambda * vec_d_lambda) where the sum is on the lambdas
P_l = np.dot(M,mu_lambda)
E =[0] * 4
i = 0
for x,y in product(domain_xy, repeat=2):
    s = 0
    for a,b in product(domain_ab, repeat=2):
        s += a *b * P[p[a,b,x,y]]
    E[i] = (-1)**(x*y) * s
    i +=1
print(f"{E = }")
print(f"CHSH = {sum(e for e in E)}")


#add a variable Q (visibility)
Q = m.addVar(name="Q", vtype="C")


#update the model with the newly defined variables
m.update()

# Add the constraints
for i in range(len(P)):
    m.addConstr(((1-Q)*P[i]  + Q*random[i] <= P_l[i])  )

m.addConstr(quicksum(mu_lambda[i] for i in range(len(lambdas))) == 1)

m.addConstr(Q<=1)
for i in range(len(lambdas)):
    m.addConstr(mu_lambda[i] >= 0)
# #
# m.addConstr(Q >= 0 )
# m.addConstr(Q <= 1 )

m.update()

# objective = Min Q
m.setObjective(Q, GRB.MINIMIZE)

m.update()

m.optimize()

# Uncomment to display the system of constraints and the objective solved by Gurobi
# m.display()

if (m.objVal > 0) :
    print("\n--------------")
    print("\n Objective value greater than 0 : NON LOCAL")

if (m.objVal == 0) :
    print("\n--------------")
    print("\n Objective value is equal to 0 :  LOCAL")



print(f"Optimal objective value S = {m.objVal}")
print(f"Solution values:     \n")
print(f"                        mu_lambda= {[mu_lambda[i].X for i in range(len(lambdas))]}")
print(f"                        Q = {Q.X }")
print(f"               (recall) P = {p}")
