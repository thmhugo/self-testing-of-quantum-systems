import numpy as np
from gurobipy import *
from game import *


def solve_primal(game, P):
    """Resolve the primal linear program.

    Args:
        game (Game): Is the description of the game.
        P (list[int]): Is the probability distribution one want to test.

    Returns:
        int: The optimal value found by Gurobi
    """
    R = np.array(uniform_noise(game))

    # mu_lambda is a vector of the coeff of the linear combination of the
    # vectors d_lambda
    mu_lambda = [game.model.addVar(name=f"mu_{i}", vtype="C") for i in range(game.N)]

    # P_l : vector  of the convex combination of the deterministic points, i.e
    # P_l = sum(mu_lambda * vec_d_lambda) where the sum is on the lambdas
    P_l = np.dot(game.M, mu_lambda)

    # add a variable Q (visibility)
    Q = game.model.addVar(name="Q", vtype="C")

    # update the model with the newly defined variables
    game.model.update()

    # Add the constraints
    for i in range(len(P)):
        game.model.addConstr(((1 - Q) * P[i] + Q * R[i] == P_l[i]))

    game.model.addConstr(gp.quicksum(mu_lambda[i] for i in range(game.N)) == 1)

    for i in range(game.N):
        game.model.addConstr(mu_lambda[i] >= 0)

    game.model.setObjective(Q, GRB.MINIMIZE)
    game.model.update()
    game.model.optimize()

    return game.model.objVal


if __name__ == "__main__":
    # CHSH
    game = Game(domain_ab=[-1, 1], domain_xy=[0, 1])
    print(solve_primal(game, quantum_probability_distribution_chsh(game=game)))
    print(solve_primal(game, uniform_noise(game)))
    # solve_chsh_primal(game,
    # [a * p + (1 - a) * 0.25 for p in quantum_probability_distribution_chsh(game=game)]
    # )

    # Mayers-Yao
    game = Game(domain_ab=[-1, 1], domain_xy=[0, 1, 2])
    print(
        solve_primal(
            game=game, P=quantum_probability_distribution_mayers_yao(game=game)
        )
    )
    print(solve_primal(game=game, P=uniform_noise(game=game)))
