from math import sqrt
from itertools import product
import collections
from gurobipy import *
import numpy as np

np.set_printoptions(threshold=np.inf)

# inputs (x,y) domain and outputs (a,b) domain
domain_xy = [0, 1, 2]
domain_ab = [-1, 1]
delta = len(domain_ab)
m = len(domain_xy)

# Cardinal of the basis of proba P(ab|xy)
N = (delta * m) ** 2

 # Stores the index of each P(a,b,x,y)
indexes_p = collections.defaultdict(int)
i = 0
for a, b in product(domain_ab, repeat=2):
    for x, y in product(domain_xy, repeat=2):
        indexes_p[a, b, x, y] = i
        i += 1

print("\n ------  basis order\n")
print(indexes_p)
print("\n\n")


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

#-------------------------------------------------------------------------------
#       LOCAL CORR
#-------------------------------------------------------------------------------

def local_corr():
    P = np.zeros(len(domain_ab)**2*len(domain_xy)**2)
    for x, y in product(domain_xy, repeat=2):
        for a, b in product(domain_ab, repeat=2):
            index = indexes_p[(a,b,x,y)]
            P[index]= 1/4.

    print("\n ------  Local Correlations")
    print(P) ;
    print("\n\n")

    return(P)

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
    for a, b in list(product(domain_ab, repeat=2)):
        for x, y in list(product(domain_xy, repeat=2)):
            dl.append(int(l[x] == a and l[y + 3] == b))
    return dl

# Lambdas are the possible outputs assignement (a0,a1,a2,b0,b1,b2), there are 64 possible lambdas

lambdas = []
for a0, a1, a2 in list(product([-1, 1], repeat=3)):
    for b0, b1, b2 in list(product([-1, 1], repeat=3)):
        lambdas.append((a0, a1, a2, b0, b1, b2))


#D_l is a matrix with each row corresponding to a deterministic behavior lambda
D_l = np.zeros((len(lambdas),N)) # vector 64*36
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
    P = ls_quantum_p()



random = np.ones(len(P))
for i in range(len(P)) :
    random[i] = 1/4.


# mu_lambda is a vector of the coeff of the linear combination of the vectors d_lambda
mu_lambda = [m.addVar(name=f"mu_{i}", vtype="C") for i in range(len(lambdas))]


# P_l : vector  of the convex combination of the deterministic points,
# i.e P_l = sum(mu_lambda * vec_d_lambda) where the sum is on the lambdas
P_l = np.dot(M,mu_lambda)

#add a variable Q (visibility)
Q = m.addVar(name="Q", vtype="C")

m.update()

# Add the constraints
for i in range(len(P)):
    m.addConstr(((1-Q)*P[i] + Q*random[i] <= P_l[i]) )

m.addConstr(quicksum(mu_lambda[i] for i in range(len(lambdas))) == 1)

for i in range(len(lambdas)):
    m.addConstr(mu_lambda[i] >= 0)

m.update()


# objective = Min Q
m.setObjective(Q, GRB.MINIMIZE)

m.update()

m.optimize()

# Uncomment to display the system of constraints and the objective solved by Gurobi
# m.display()

if (m.objVal > 0) :
    print("\n--------------")
    print("\n Objective value greater than zero : NON LOCAL")

if (m.objVal == 0) :
    print("\n--------------")
    print("\n Objective value is equal to one :  LOCAL")

check = np.zeros(len(P))

for i in range(len(lambdas)):
    check += mu_lambda[i].X * np.array(vec_d_lambda(lambdas[i]))

print("check")
print(check)


print(f"Optimal objective value S = {m.objVal}")
print(f"Solution values:     \n")
print(f"                        mu_lambda= {[mu_lambda[i].X for i in range(len(lambdas))]}")
print(f"                        Q = {Q.X }")
print(f"               (recall) P = {P}")
