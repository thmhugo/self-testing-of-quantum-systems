from math import sqrt
import gurobipy as gp
import numpy as np
from game import *


domain_xy = [0, 1]
domain_ab = [-1, 1]

game = Game(domain_xy, domain_ab)

# M is the transpose of D_l , one column = one deterministic behavior d_lambda
M = np.column_stack(game.d_lambda)


def solve_chsh_dual(game, P):
    """_summary_

    Args:
        game (_type_): _description_
        P (_type_): _description_
    """
    R = np.array(uniform_noise(game))

    # Create variables
    Y = [game.model.addVar(name=f"y_{i}", vtype="C") for i in range(game.N)]
    gamma_p = game.model.addVar(name="gamma_p", vtype="C")
    gamma_m = game.model.addVar(name="gamma_m", vtype="C")

    game.model.update()

    # Set objective function
    game.model.setObjective(gurobi_dot(P, Y) + gamma_p - gamma_m, gp.GRB.MAXIMIZE)

    # Add constraints
    for l in game.lambdas:
        game.model.addConstr(
            gamma_p - gamma_m + gurobi_dot(Y, game.vec_d_lambda(l)) <= 0
        )

    game.model.addConstr(
        gp.quicksum((-R[i] + P[i]) * (Y[i]) for i in range(len(P))) <= 1
    )
    game.model.update()
    game.model.optimize()

    print(f"Optimal objective value S = {game.model.objVal}")
    print(f"Solution values:")
    print(f"               (recall) P = \n{np.array(P)}")
    print(f"                   Y      = \n{np.array([Y[i].X for i in range(game.N)])}")
    print(f"                  gamma_p = {gamma_p.X }")
    print(f"                  gamma_m = {gamma_m.X }")

    L = [y.X for y in Y]
    print("R•Y : ", gurobi_dot(L, R))
    print("P•Y : ", gurobi_dot(P, L))
    print("Y•Y : ", gurobi_dot(L, L))


# solve_chsh_dual(game, ns_p())
solve_chsh_dual(game, quantum_probability_distribution_chsh(game=game))
