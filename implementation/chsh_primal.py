from math import sqrt
import numpy as np
from itertools import product
from gurobipy import *
from game import *

# inputs (x,y) domain and outputs (a,b) domain
domain_xy = [0, 1]
domain_ab = [-1, 1]

game = Game(domain_ab=domain_ab, domain_xy=domain_xy)

# M is the transpose of d_lambda , one column = one deterministic behavior
# d_lambda
M = np.column_stack(game.d_lambda)


def solve_chsh_primal(game, P):
    R = np.array(uniform_noise(game))

    # mu_lambda is a vector of the coeff of the linear combination of the
    # vectors d_lambda
    mu_lambda = [game.model.addVar(name=f"mu_{i}", vtype="C") for i in range(game.N)]

    # P_l : vector  of the convex combination of the deterministic points, i.e
    # P_l = sum(mu_lambda * vec_d_lambda) where the sum is on the lambdas
    P_l = np.dot(M, mu_lambda)
    E = [0] * 4
    i = 0
    for x, y in product(domain_xy, repeat=2):
        s = 0
        for a, b in product(domain_ab, repeat=2):
            s += a * b * P[game.indexes_p[a, b, x, y]]
        E[i] = (-1) ** (x * y) * s
        i += 1

    # add a variable Q (visibility)
    Q = game.model.addVar(name="Q", vtype="C")

    # update the model with the newly defined variables
    game.model.update()

    # Add the constraints
    for i in range(len(P)):
        game.model.addConstr(((1 - Q) * P[i] + Q * R[i] >= P_l[i]))

    game.model.addConstr(gp.quicksum(mu_lambda[i] for i in range(game.N)) == 1)

    for i in range(game.N):
        game.model.addConstr(mu_lambda[i] >= 0)

    game.model.setObjective(Q, GRB.MINIMIZE)
    game.model.update()
    game.model.optimize()

    print("            E = ", np.array(E))
    print("         CHSH = ", sum(e for e in E))
    print("Optimal value = ", game.model.objVal)

    if game.model.objVal == 0:
        print("LOCAL")
    else:
        print("NON LOCAL")


solve_chsh_primal(game, quantum_probability_distribution_chsh(game=game))
