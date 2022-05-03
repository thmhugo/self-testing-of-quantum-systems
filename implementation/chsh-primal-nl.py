from math import sqrt, asin
import numpy as np
from itertools import product
import collections
from gurobipy import *
from decimal import Context
from gurobipy import *
from game import *


def get_model(game):
    game = Game(domain_ab=game.domain_ab, domain_xy=game.domain_xy)
    game.model.Params.LogToConsole = 0
    game.model.params.NonConvex = 2  # Needed since the model is non convex.

    # mu_lambda is a vector of the coeff of the linear combination of the vectors d_lambda
    mu_lambda = [
        game.model.addVar(name=f"mu_{i}", vtype="C") for i in range(len(game.lambdas))
    ]

    P = [game.model.addVar(name=f"p_{i}", vtype="C") for i in range(16)]
    # P_l : vector  of the convex combination of the deterministic points,
    # i.e P_l = sum(mu_lambda * vec_d_lambda) where the sum is on the lambdas
    P_l = np.dot(game.M, mu_lambda)

    # add a variable Q (visibility)
    Q = game.model.addVar(name="Q", vtype="C")

    # update the model with the newly defined variables
    game.model.update()

    # Add the constraints
    for i in range(len(P)):
        game.model.addConstr(((1 - Q) * P[i] + Q - P_l[i] == 0))

    # Constraints on probabilities
    game.model.addConstr(quicksum(mu_lambda[i] for i in range(len(game.lambdas))) >= 1)

    for i in range(len(game.lambdas)):
        game.model.addConstr(mu_lambda[i] >= 0)
        game.model.addConstr(P[i] >= 0)

    # Define a constraint for the chsh inequality
    chsh = LinExpr()
    for x, y in product(game.domain_xy, repeat=2):
        e = LinExpr()  # represents an expecation value <Ax By>
        for a, b in product(game.domain_ab, repeat=2):
            e += a * b * P[game.indexes_p[a, b, x, y]]
        # Constraints from (10) p.6 : Bell nonlocality
        game.model.addConstr(e <= 1)
        game.model.addConstr(e >= -1)
        chsh += (-1) ** (x * y) * e

    for x, y in product(game.domain_xy, repeat=2):
        # Each marginal must sum up to one
        game.model.addConstr(
            quicksum(
                P[game.indexes_p[a, b, x, y]]
                for a, b in product(game.domain_ab, repeat=2)
            )
            == 1
        )
    # If useless, removed by guroby
    # m.addConstr(quicksum(P[i] for i in range(len(lambdas))) == 4)

    for x, y in product(game.domain_xy, repeat=2):
        game.model.addConstr(
            quicksum(
                a * P[game.indexes_p[a, b, x, y]]
                for a, b in product(game.domain_ab, repeat=2)
            )
            == 0
        )

    for x, y in product(game.domain_xy, repeat=2):
        game.model.addConstr(
            quicksum(
                b * P[game.indexes_p[a, b, x, y]]
                for a, b in product(game.domain_ab, repeat=2)
            )
            == 0
        )

    game.model.addConstr(Q >= 0)
    game.model.addConstr(Q <= 1)
    return game, chsh, P, Q


def find_optimal_p(game):
    run = True
    d = 10e-4
    upper_bound = 0

    c = Context(prec=15)

    while run:
        game, chsh, P, Q = get_model(game)
        bound = game.model.addConstr(chsh <= 2.5 + upper_bound)

        game.model.update()
        game.model.setObjective(chsh, GRB.MAXIMIZE)
        game.model.update()

        game.model.optimize()
        game.model.remove(bound)
        upper_bound += d

        chsh_check = 0
        for x, y in product(game.domain_xy, repeat=2):
            e = 0
            for a, b in product(game.domain_ab, repeat=2):
                e += a * b * P[game.indexes_p[a, b, x, y]].X
            chsh_check += (-1) ** (x * y) * asin(e)

        print(
            f"check : {c.create_decimal(abs(chsh_check))} CHSH value : {c.create_decimal(chsh.getValue())}            ",
            end="\r",
        )
        run = chsh.getValue() <= 2 * sqrt(2)
        # input()

    print(f"Objective = {game.model.objVal}")
    print(f"{Q.X = }")
    print(f"Computed probability distribution : \n{([P[i].X for i in range(16)])}")
    chsh_check = 0
    for x, y in product(game.domain_xy, repeat=2):
        e = 0
        for a, b in product(game.domain_ab, repeat=2):
            e += a * b * P[game.indexes_p[a, b, x, y]].X
        chsh_check += (-1) ** (x * y) * asin(e)

    print(
        f"check : {c.create_decimal(abs(chsh_check))} CHSH value : {c.create_decimal(chsh.getValue())}            ",
    )


def using_bit_mask(game):

    use_qp_mask = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]

    P = [
        0.42677669529663687,
        0.07322330470336313,
        0.07322330470336313,
        0.42677669529663687,
        0.42677669529663687,
        0.07322330470336313,
        0.07322330470336313,
        0.42677669529663687,
        0.42677669529663687,
        0.07322330470336313,
        0.07322330470336313,
        0.42677669529663687,
        0.07322330470336313,
        0.42677669529663687,
        0.42677669529663687,
        0.07322330470336313,
    ]

    m, chsh, _, Q = get_model(game)

    previous_chsh = 0
    run = True
    d = 10e-4
    upper_bound = 0

    c = Context(prec=15)

    while run:
        game, chsh, P, Q = get_model(game)
        qp = quantum_probability_distribution_chsh(game)

        for i in range(16):
            if use_qp_mask[i]:
                game.model.addConstr(P[i] == qp[i])

        bound = game.model.addConstr(chsh >= 2 - upper_bound)

        game.model.update()
        game.model.setObjective(chsh, GRB.MINIMIZE)
        game.model.update()

        game.model.optimize()
        game.model.remove(bound)
        upper_bound += d

        chsh_check = 0
        for x, y in product(game.domain_xy, repeat=2):
            e = 0
            for a, b in product(game.domain_ab, repeat=2):
                e += a * b * P[game.indexes_p[a, b, x, y]].X
            chsh_check += (-1) ** (x * y) * asin(e)

        print(
            f"check : {c.create_decimal(abs(chsh_check))} CHSH value : {c.create_decimal(chsh.getValue())}            ",
            end="\r",
        )
        run = chsh.getValue() >= 1 and previous_chsh != chsh.getValue()
        previous_chsh = chsh.getValue()
    print(chsh.getValue())

    print(f"Objective = {game.model.objVal}")
    print(f"{Q.X = }")
    print(f"Computed probability distribution : \n{([P[i].X for i in range(16)])}")
    chsh_check = 0
    for x, y in product(game.domain_xy, repeat=2):
        e = 0
        for a, b in product(game.domain_ab, repeat=2):
            e += a * b * P[game.indexes_p[a, b, x, y]].X
        chsh_check += (-1) ** (x * y) * asin(e)

    print(
        f"check : {c.create_decimal(abs(chsh_check))} CHSH value : {c.create_decimal(chsh.getValue())}            ",
    )


if __name__ == "__main__":
    chsh_game = Game(domain_ab=[-1, 1], domain_xy=[0, 1])
    # find_optimal_p(chsh_game)
    using_bit_mask(chsh_game)
