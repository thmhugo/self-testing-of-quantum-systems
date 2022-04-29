from math import sqrt
from itertools import product
import collections
from xml import dom
import gurobipy as gp
import numpy as np
from game import *

np.set_printoptions(threshold=np.inf)

game = Game(domain_ab=[-1, 1], domain_xy=[0, 1, 2])


p = quantum_probability_distribution_mayers_yao(game)


def solve_mayers_yao_dual(game, p):
    R = np.ones(len(p))
    for i in range(len(p)):
        R[i] = 1 / 4.0

    # Create a new model
    m = gp.Model()
    m.Params.LogToConsole = 0  # Less verbose Guroby output.

    Y = [m.addVar(name=f"y_{i}", vtype="C") for i in range(N)]

    # Create variables

    gamma_p = m.addVar(name="gamma_p", vtype="C")
    gamma_m = m.addVar(name="gamma_m", vtype="C")

    m.update()

    # Set objective function
    m.setObjective(gurobi_dot(p, Y) + gamma_p - gamma_m, gp.GRB.MAXIMIZE)

    # Add constraints

    for l in lambdas:
        m.addConstr(gamma_p - gamma_m + gurobi_dot(Y, vec_d_lambda(l)) <= 0)

    m.addConstr(gp.quicksum((-R[i] + p[i]) * (Y[i]) for i in range(len(p))) <= 1)

    m.update()
    # Solve it!
    m.optimize()
    # m.display()

    print(f"Optimal objective value S = {m.objVal}")
    print(f"Solution values:     \n")
    print(f"                        Y = {[Y[i].X for i in range(N)]}")
    print(f"                        gamma_p = {gamma_p.X }")
    print(f"                        gamma_m = {gamma_m.X }")

    print(f"               (recall) P = {p}")

    print("dot R Y : ")
    L = [(Y[i].X) for i in range(N)]
    print(gurobi_dot(L, R))
    print("dot P Y : ")
    print(gurobi_dot(p, L))

    print("dot Y Y : ")
    print(gurobi_dot(L, L))

    # evaluated_gurobi_dot = lambda a, b: sum(a[i].X * b[i] for i in range(len(b)))
    #
    # print(f"Inequality : s • p = {evaluated_gurobi_dot(Y, p)}" )


solve_mayers_yao_dual(game, p)
