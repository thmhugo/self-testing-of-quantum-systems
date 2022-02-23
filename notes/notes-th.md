---
header-includes: |
	\usepackage{amssymb}
	\usepackage{bbm}
	\usepackage{braket}
	\usepackage{mathrsfs}
---
# The self-testing scenario

$\mathscr{L(H)}$ denotes the set of linear operators acting on Hilbert space
$\mathscr H$.

We know there exist measurement operators $M_{a|x} \in \mathscr{L(H)}$ acting on
Alice's Hilbert space and satisfying

$$M_{a|x} \succcurlyeq 0 ; \forall a, x \sum_a M_{a|x} = Id_A$$

Similarly there exist measurement operators $N_{b|y} \in \mathscr{L(H)}$ acting
on Bob's Hilbert space. The measurement operators are therefore projective :

$$\forall a, a' : \quad M_{a|x}M_{a'|x} = \delta_{a, a'}M_{a|x}$$

$$\forall b, b' : \quad N_{b|y}N_{b'|y} = \delta_{b, b'}N_{b|y}$$

Now, from the Born rule, there must exist some quantum state $\rho_{AB} \in
\mathscr{L(H_\text{A} \otimes H_\text{B})} \succcurlyeq 0$, and tr$\rho_{AB}$ =
1 such that

$$p(a,b|x,y) = \text{tr}\big[\rho_{AB} M_{a|x} \otimes N_{b|y} \big]$$

In self-testing, one aims to infer the form of the state and the measurement in
the trace from knowledge of the correlation $p(a, b|x, y)$ alone, i.e. in
device-independent scenario.

_Born rule_ : A key postulate of quantum mechanics which gives the probability
that a measurement of a quantum system will yield a given result. More formally,
for a state $\ket{\psi}$ and a $F_i$ POVM element (associated with the measurement
outcome $i$), then the probability of obtaining $i$ when measuring $\ket{\psi}$ is
given by

$$p(i) = \braket{\psi | F_i | \psi}$$

## Physical assumptions

1. The experiment admits a quantum description (state and measurement)
2. The laboratories of Alice and Bob are located in separate location in space
   and there is no communication between the two laboratories.
3. The setting $x$ and $y$ are chosen freely and independently of all other
   systems in the experiment.
4. Each round of the experiment is independent of all other rounds a physically
   equivalent to all others (i.e. there exists a single density matrix and
   measurement operators that are valid in every round).

## Impossibility to infer exactly the references

1. _Unitary invariance of the trace_ : one can reproduce the statistics of any
   state $\ket{\psi}$ and measurement $\{M_{a|x}\}, \{N_{b|y}\}$ by instead using
   the rotated state $U \otimes V \ket{\psi}$ and measurement
   $\{UM_{a|x}U^\dagger\}, \{VN_{b|y}V^\dagger\}$, where $U, V$ are unitary
   transformations. Hence, one can never conclude that the state was $\ket{\psi}$
   or $U \otimes V \ket{\psi}$.

2. _Additional degrees of freedom_ : a state $\ket{\psi} \otimes \ket \xi$ and
   measurements $\{M_{a|x} \otimes Id_\xi\}, \{N_{b|y} \otimes Id_\xi\}$ gives
   the same correlation as $\ket{\psi}$ and $\{M_{a|x}\}, \{N_{b|y}\}$.
