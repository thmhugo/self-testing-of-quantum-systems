from math import sqrt
from itertools import product
import collections
from gurobipy import *
import numpy as np
from sympy import solve
from game import *

np.set_printoptions(threshold=np.inf)

# inputs (x,y) domain and outputs (a,b) domain

# Cardinal of the basis of proba P(ab|xy)

# M is the transpose of D_l , one column = one deterministic behavior d_lambda

game = Game(domain_ab=[-1, 1], domain_xy=[0, 1, 2])
M = np.column_stack(game.d_lambda)


def solve_mayers_yao_primal(game, P):
    R = np.array(uniform_noise(game))

    # mu_lambda is a vector of the coeff of the linear combination of the vectors d_lambda
    mu_lambda = [game.model.addVar(name=f"mu_{i}", vtype="C") for i in range(game.N)]

    # P_l : vector  of the convex combination of the deterministic points,
    # i.e P_l = sum(mu_lambda * vec_d_lambda) where the sum is on the lambdas
    P_l = np.dot(M, mu_lambda)

    # add a variable Q (visibility)
    Q = game.model.addVar(name="Q", vtype="C")

    game.model.update()

    # Add the constraints
    for i in range(len(P)):
        game.model.addConstr(((1 - Q) * P[i] + Q * R[i] == P_l[i]))

    game.model.addConstr(gp.quicksum(mu_lambda[i] for i in range(game.N)) == 1)

    for i in range(game.N):
        game.model.addConstr(mu_lambda[i] >= 0)

    game.model.update()

    # objective = Min Q
    game.model.setObjective(Q, GRB.MINIMIZE)
    game.model.update()
    game.model.optimize()

    # Uncomment to display the system of constraints and the objective solved by Gurobi
    # m.display()
    print(game.model.objVal)

    if game.model.objVal > 0:
        print("\n--------------")
        print("\n Objective value greater than one : NON LOCAL")

    if game.model.objVal == 0:
        print("\n--------------")
        print("\n Objective value is equal to one :  LOCAL")


solve_mayers_yao_primal(game=game, P=quantum_probability_distribution_mayers_yao(game=game))
# solve_mayers_yao_primal(game=game, P=uniform_noise(game=game))
