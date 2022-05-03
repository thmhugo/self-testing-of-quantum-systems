from math import sqrt
from itertools import product
import collections
from gurobipy import *
import numpy as np
from sympy import EvaluationFailed
from game import *


def solve_dual(game, P):
    """_summary_

    Args:
        game (_type_): _description_
        P (_type_): _description_

    Returns:
        _type_: _description_
    """
    R = np.array(uniform_noise(game))
    N = (game.delta * game.m) ** 2  # Could be stored in Game.
    # Create variables
    Y = [game.model.addVar(name=f"y_{i}", vtype="C") for i in range(N)]
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
    print(f"                   Y      = \n{np.array([Y[i].X for i in range(N)])}")
    print(f"                  gamma_p = {gamma_p.X }")
    print(f"                  gamma_m = {gamma_m.X }")

    L = [y.X for y in Y]
    print("R•Y :", evaluated_gurobi_dot(L, R))
    print("P•Y :", evaluated_gurobi_dot(P, L))
    print("Y•Y :", evaluated_gurobi_dot(L, L))

    return game.model.objVal


if __name__ == "__main__":
    game = Game(domain_xy=[0, 1], domain_ab=[-1, 1])
    solve_dual(game, quantum_probability_distribution_chsh(game=game))
    print("----------")
    game = Game(domain_ab=[-1, 1], domain_xy=[0, 1, 2])
    solve_dual(game, quantum_probability_distribution_mayers_yao(game=game))
