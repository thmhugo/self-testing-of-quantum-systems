---
header-includes: |
      \usepackage{amssymb}
      \usepackage{bbm}
      \usepackage{braket}
      \usepackage{mathrsfs}
geometry:
 - left=1cm
 - right=1cm
 - top=1cm
 - bottom=25 mm
---
# The self-testing scenario

$\mathscr{L(H)}$ denotes the set of linear operators acting on Hilbert space
$\mathscr H$.

We know there exist measurement operators $M_{a|x} \in \mathscr{L(H)}$ acting on
Alice's Hilbert space and satisfying
$$M_{a|x} \succcurlyeq 0 ; \forall a, x \sum_a M_{a|x} = Id_A$$
{#eq:measurement-properties}

Similarly there exist measurement operators $N_{b|y} \in \mathscr{L(H)}$ acting
on Bob's Hilbert space. The measurement operators are therefore projective :
$$\forall a, a' : \quad M_{a|x}M_{a'|x} = \delta_{a, a'}M_{a|x}$${#eq:_}
$$\forall b, b' : \quad N_{b|y}N_{b'|y} = \delta_{b, b'}N_{b|y}$${#eq:__}

Now, from the Born rule, there must exist some quantum state $\rho_{AB} \in
\mathscr{L(H_\text{A} \otimes H_\text{B})} \succcurlyeq 0$, and tr$\rho_{AB}$ =
1 such that
$$p(a,b|x,y) = \text{tr}\big[\rho_{AB} M_{a|x} \otimes N_{b|y} \big]$$ {#eq:_}

In self-testing, one aims to infer the form of the state and the measurement in
the trace from knowledge of the correlation $p(a, b|x, y)$ alone, i.e. in
device-independent scenario.

_Born rule_ : A key postulate of quantum mechanics which gives the probability
that a measurement of a quantum system will yield a given result. More formally,
for a state $\ket{\psi}$ and a $F_i$ POVM element (associated with the
measurement outcome $i$), then the probability of obtaining $i$ when measuring
$\ket{\psi}$ is given by $$p(i) = \braket{\psi | F_i | \psi}$$ {#eq:_}

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
or $U \otimes V \ket{\psi}$. \newline On the other hand, considering real
reference states $(\ket{\psi} = \ket{\psi}^*)$, one can only self-test
measurements that are invariant under the complex conjugate $*$, since, assuming
a real state $\ket{\psi}$, $p(ab|xy) = \text{tr}\big[\ket{\psi}\bra{\psi}
M_{a|x} \otimes N_{b|y} \big] = \text{tr}\big[\ket{\psi}\bra{\psi} M_{a|x}^*
\otimes N_{b|y}^* \big]$. Thus any correlation obtained using $\big\{\ket{\psi},
M_{a|x}, N_{b|y} \big\}$ can also be obtained using $\big\{\ket{\psi},
M_{a|x}^*, N_{b|y}^* \big\}$; but the second is not related to the first one via
a local isometry (It's an open problem to list the set of state and measuremend
transformation that do not affect the probabilities).

2. _Additional degrees of freedom_ : a state $\ket{\psi} \otimes \ket \xi$ and
measurements $\{M_{a|x} \otimes Id_\xi\}, \{N_{b|y} \otimes Id_\xi\}$ gives the
same correlation as $\ket{\psi}$ and $\{M_{a|x}\}, \{N_{b|y}\}$.

\newpage

## Extractability relative to a Bell Inequality

The extractability $\Xi$ is defined as the maximum fidelity of $\Lambda_A
\otimes \Lambda_B \big[\rho\big]$ and $\ket{\psi'}$ over all CPTP (Completely
positive and trace preserving) maps: $$ \Xi (\rho \rightarrow \ket{\psi'}) =
\text{max}_{\Lambda_A ,\Lambda_B} F(\Lambda_A \otimes \Lambda_B, \ket{\psi'})$$
{#eq:extractability}
where $\rho \rightarrow \ket{\psi'}$ defines a kind of mapping of the test state
$\rho$ to the target state $\ket{\psi'}$. The maximum is taken over all quantum
channels (why are the $\Lambda_{A,B}$ called _quantum channels_ ?). This
implies that $\Xi$ return the $\Lambda_{A,B}$ such that the fidelity to the
reference state is maximal.

In order to test the entanglement characteristics of $\rho$, $\ket{\psi'}$ is
assumed to be a state which achieves the maximal quantum violation. Hence, when
the maximal quantum violation is observed in a self-testing scenario, the shared
unknown state $(\rho)$ can be mapped to $\ket{\psi'}$, and the resulting
extractability is 1.

To get the optimal (robustness-wise) self-testing statement, one can minimize
the possible extractability (over all states) when a violation of at least
$\beta$ is observed on a Bell inequality $B$. This quantity can be captured by
the function $\mathcal{Q}$ defined as
$$ \mathcal{Q}_{\psi,\mathcal{B}_\mathcal{I}} = \text{min}_{\rho \in
S_\mathcal{B}(\beta)} \quad \Xi (\rho \rightarrow \ket{\psi'}) $$
{#eq:robustness}

where $S_\mathcal{B}(\beta)$ is the set of states $\rho$ which
violate Bell inequality $\mathcal B$ with value at least $\beta$.  One needs to
note that the optimal CPTP map generally depends on the observed violation.
