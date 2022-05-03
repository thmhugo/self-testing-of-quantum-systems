from cProfile import label
from math import sqrt
from gurobipy import *
from game import *
from primal import solve_primal
from decimal import Context
import tikzplotlib


noisy_p = lambda a, P: [a * p + (1 - a) * 0.25 for p in P]


def search_minimum_a(game, P, prec=20):
    context = Context(prec=prec)
    upper_bound = 1
    lower_bound = 0
    last_value = 0
    n = 0

    while True:
        n += 1
        a = (upper_bound + lower_bound) / 2.0
        obj = solve_primal(P=noisy_p(a, P), game=game)

        if obj > 0:  # Quantum probability distribution
            upper_bound = a
        else:
            lower_bound = a

        if abs(last_value - context.create_decimal(a)) < 10 ** (-prec):
            break  # Quite a bad way to exit a loop

        last_value = context.create_decimal(a)

    return (n, a)


if __name__ == "__main__":
    chsh_game = Game(domain_ab=[-1, 1], domain_xy=[0, 1])
    my_game = Game(domain_ab=[-1, 1], domain_xy=[0, 1, 2])
    chsh_P = quantum_probability_distribution_chsh(chsh_game)
    my_P = quantum_probability_distribution_mayers_yao(my_game)

    ns_P = no_signaling_probability_distribution_chsh(chsh_game)
    print(chsh_value(chsh_game, ns_P))
    n, a = search_minimum_a(chsh_game, chsh_P)

    print(f"[CHSH] Iterated {n} times before stopping.")
    print(f"[CHSH]             {a = }")
    print(f"[CHSH]   {1 / sqrt(2) = }\n")
    print(solve_primal(chsh_game, noisy_p(a, chsh_P)))  # "Local" behavior
    print(
        solve_primal(chsh_game, noisy_p(a + 10e-5, chsh_P))
    )  # Something just above the local one

    n, a = search_minimum_a(my_game, my_P)

    print(f"[MY] Iterated {n} times before stopping.")
    print(f"[MY]             {a = }")

    print(solve_primal(my_game, noisy_p(a, my_P)))  # "Local" behavior
    print(
        solve_primal(my_game, noisy_p(a + 10e-5, my_P))
    )  # Something just above the local one

    # We could draw stuff with this :
    a = 1
    step = 10e-3

    A = []
    chsh_obj, my_obj = [], []
    chsh = 3

    while a > 0:
        # obj =
        chsh = chsh_value(chsh_game, noisy_p(a, chsh_P))
        A.append(a)
        chsh_obj.append(solve_primal(P=noisy_p(a, chsh_P), game=chsh_game))
        my_obj.append(solve_primal(P=noisy_p(a, my_P), game=my_game))
        print(chsh, end="\r")
        a -= step

    import matplotlib.pyplot as plt

    plt.plot(A, my_obj, c="m", label="MY")
    plt.plot(
        search_minimum_a(my_game, my_P)[1],
        0,
        "mo",
        label=r"$\alpha \approx 0.82$",
    )

    plt.plot(A, chsh_obj, c="b", label="CHSH")
    plt.plot(
        search_minimum_a(chsh_game, chsh_P)[1],
        0,
        "bo",
        label=r"$\alpha = \frac{1}{\sqrt{2}}$",
    )

    plt.xlabel(r"accuracy $\alpha$")
    plt.ylabel("primal objective")

    plt.legend()

    # plt.show()
    tikzplotlib.save("../notes/noisy-channel.tex")
