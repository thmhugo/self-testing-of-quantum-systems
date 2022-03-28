from math import sqrt
from itertools import product
import gurobipy as gp


def vec_d_lambda(l):
    dl = []
    # permutations = list(product([0, 1], repeat=2))
    for x, y in list(product([0, 1], repeat=2)):
        for a, b in list(product([-1, 1], repeat=2)):
            # print(f"l= {l} ; a={a}, b={b}, x={x}, y={y+2}")
            dl.append(int(l[x] == a and l[y + 2] == b))
    return dl


# lambdas = list(product([0, 1], repeat=4))
lambdas = []
for a, b in list(product([-1, 1], repeat=2)):
    for x, y in list(product([0, 1], repeat=2)):
        lambdas.append((a, b, x, y))

for l in lambdas:
    print(f"{l} -> {vec_d_lambda(l)}")

dot = lambda A, B: gp.quicksum(A[i] * B[i] for i in range(N))
prob = lambda a, b, x, y: int((a + b) % 2 == x * y) * 0.5
rescale = lambda a: int(a + 1) / 2

delta = 2
m = 2

N = delta ** (2 * m)
p = [prob(a, b, x, y) for a, b, x, y in list(product([0, 1], repeat=4))]

minus = (sqrt(2) - 1) / (4 * sqrt(2))
plus = (1 + sqrt(2)) / (4 * sqrt(2))

p = [  # (a, b, x, y)
    plus,  #  (-1, -1, 0, 0)
    plus,  #  (-1, -1, 0, 1)
    plus,  #  (-1, -1, 1, 0)
    minus,  # (-1, -1, 1, 1)
    minus,  # (-1,  1, 0, 0)
    minus,  # (-1,  1, 0, 1)
    minus,  # (-1,  1, 1, 0)
    plus,  #  (-1,  1, 1, 1)
    minus,  #  ( 1, -1, 0, 0)
    plus,  #  ( 1, -1, 0, 1)
    minus,  # ( 1, -1, 1, 0)
    minus,  # ( 1, -1, 1, 1)
    plus,  #  ( 1,  1, 0, 0)
    plus,  #  ( 1,  1, 0, 1)
    plus,  #  ( 1,  1, 1, 0)
    minus,  # ( 1,  1, 1, 1)
]

print(p)
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


# m.addConstr(gp.quicksum([s for s in S]) == 1)
# for s in S:
#     m.addConstr(s >= 0)

# Solve it!
m.optimize()

print(f"Optimal objective value = {m.objVal}")
print(f"Solution values:    S_l = {S_l.X}")
print(f"                      S = {[S[i].X for i in range(N)]}")
print(f"             (recall) P = {p}")
dot_p = lambda A, B: gp.quicksum(A[i].X * B[i] for i in range(N))

print(dot_p(S, p))
# chsh = 0
# for a,b in product([-1, 1],repeat=2):
#     chsh += a*b*prob(rescale(a),_b,0,0)
#     chsh += a*b*prob(_a,_b,0,1)
#     chsh += a*b*prob(_a,_b,1,0)
#     chsh -= a*b*prob(_a,_b,1,1)

# print (chsh)
