from itertools import product
import gurobipy as gp


def vec_d_lambda(l):
    dl = []
    permutations = list(product([0, 1], repeat=2))

    for (x, y) in permutations:
        for (a, b) in permutations:
            dl.append(int(l[x] == a and l[y + 2] == b))

    return dl


lambdas = list(product([0, 1], repeat=4))

# for l in lambdas:
#     print(f"{l} -> {vec_d_lambda(l)}")

dot = lambda A, B: gp.quicksum(A[i] * B[i] for i in range(N))

delta = 2
m = 2

N = delta ** (2 * m)
p = [int((a + b) % 2 == x * y) for a, b, x, y in list(product([0, 1], repeat=4))]

# Create a new model
m = gp.Model()


# Create variables
S = [m.addVar(name=f"s_{i}", vtype="C") for i in range(N)]
S_l = m.addVar(name="S_l", vtype="C")
m.update()


# Set objective function
m.setObjective(dot(S, p) - S_l, gp.GRB.MAXIMIZE)


# Add constraints
for l in lambdas:
    m.addConstr(dot(S, vec_d_lambda(l)) - S_l <= 0)

m.addConstr(dot(S, p) - S_l <= 1)


# Solve it!
m.optimize()

print(f"Optimal objective value = {m.objVal}")
print(f"Solution values:    S_l = {S_l.X}")
print(f"                      S = {[S[i].X for i in range(N)]}")
print(f"             (recall) p = {p}")
