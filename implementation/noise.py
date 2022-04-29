from math import sqrt
from operator import is_
import numpy as np
from itertools import product
from gurobipy import *
from game import *
from chsh_primal import solve_chsh_primal
from decimal import Context

# inputs (x,y) domain and outputs (a,b) domain
domain_xy = [0, 1]
domain_ab = [-1, 1]

game = Game(domain_ab=domain_ab, domain_xy=domain_xy)

# M is the transpose of D_l , one column = one deterministic behavior d_lambda
M = np.column_stack(game.d_lambda)


def is_p_quantum(game, P):
    """_summary_

    Args:
        game (_type_): _description_
        P (_type_): _description_

    Returns:
        True: if the given behavior is local
    """
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

    return (not (game.model.objVal == 1), sum(e for e in E))


noisy_p = lambda a, P: [a * p + (1 - a) * 0.25 for p in P]

P = quantum_probability_distribution_chsh(game)

prec = 14
context = Context(prec=prec)
upper_bound = 1
lower_bound = 0
last = 0
n = 0

# Dichotomic search for 'a'
while True:
    n += 1
    a = (upper_bound + lower_bound) / 2.0
    is_q, chsh = is_p_quantum(P=noisy_p(a, P), game=game)
    if chsh > 2:
        upper_bound = a
    else:
        lower_bound = a

    if abs(last - context.create_decimal(a)) < 10 ** (-prec):
        break
    last = context.create_decimal(a)

if __name__ == "__main__":
    print(f"Iterated {n} times before stopping.")
    print(f"            {a = }")
    print(f"  {1 / sqrt(2) = }\n")
    solve_chsh_primal(game, noisy_p(a, P))  # Local behavior
    solve_chsh_primal(game, noisy_p(a + 10e-3, P))  # Something just above the local one
