import numpy as np
import quimb.tensor as qtn
import scipy.linalg as spla
import re
from dataclasses import dataclass

Px = np.array([[0, 1], [1, 0]])
Py = np.array([[0, -1j], [1j, 0]])
Pz = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])


@dataclass
class TwoTermsOrder1:
    # two terms, order, p = 1:
    # (m = 2) Not optimized
    a1 = 1.
    a2 = 1.
    a_list: list = (a1, a2)
    pattern = 0


@dataclass
class TwoTermsOrder2:
    # two terms, order, p = 2:
    # (m = 5, type S, ν = 3). – There are two constraints and hence one free parameter; we
    # choose a1. The error is minimized for
    # b1 = 1/2, a2 =1−2a1, a1 = 16(3−√3)≈0.21132 A< B
    # Type S: symmetic
    b1 = 1 / 2,
    a1 = 1 / 6 * (3 - np.sqrt(3))
    a2 = 1 - 2 * a1
    a_list: list = (a1, b1, a2, b1, a1)
    pattern = 0


@dataclass
class TwoTermsOrder4:
    # Optimized (m = 9, type S, ν = 5). – There are four constraints and hence there is one free parameter;
    # according to the Gröbner basis, we can choose b1. The error is minimized for
    # b2 = 1/2 − b1, a3 = 1 − 2(a1 + a2), b1 = −0.35905925216967795307,
    # a1 = 0.26756486526206148829, a2 = −0.034180403245134195595  (B < A)

    a1 = 0.26756486526206148829
    a2 = -0.034180403245134195595
    b1 = -0.35905925216967795307
    b2 = 1 / 2 - b1
    a3 = 1 - 2 * (a1 + a2)
    a_list = (a1, b1, a2, b2, a3, b2, a2, b1, a1)
    pattern = 1


@dataclass
class TwoTermsOrder6:
    # Order p = 6 - (m = 19, type SL, ν = 5).
    w1 = 0.18793069262651671457
    w2 = 0.5553
    w3 = 0.12837035888423653774
    w4 = -0.84315275357471264676
    w5 = 1 - 2 * (w1 + w2 + w3 + w4)
    half_list = [1 / 2 * w1, w1, 1 / 2 * (w1 + w2), w2, 1 / 2 * (w2 + w3), w3, 1 / 2 * (w3 + w4), w4,
                 1 / 2 * (w4 + w5),
                 w5]
    a_list = tuple(half_list + half_list[:-1][::-1])
    pattern = 0


# Number of layers of bricks per step
trotter_schemes_per_order = {1: TwoTermsOrder1,
                             2: TwoTermsOrder2,
                             4: TwoTermsOrder4,
                             6: TwoTermsOrder6}


def trotter_evolution_optimized_nn_ising_tn(L: int, Jxx: float, Jz: float, Jx: float, dt: float, steps: int,
                                            p: int = 2):  # (B < A) so set B = XX+X, A = Z
    if p == 1:
        a_list = TwoTermsOrder1.a_list
        pattern = TwoTermsOrder1.pattern
    elif p == 2:
        a_list = TwoTermsOrder2.a_list
        pattern = TwoTermsOrder2.pattern
    elif p == 4:
        a_list = TwoTermsOrder4.a_list
        pattern = TwoTermsOrder4.pattern
    elif p == 6:
        a_list = TwoTermsOrder6.a_list
        pattern = TwoTermsOrder6.pattern
    else:
        raise NotImplementedError
    leg_indx = 0
    index_list = [f'k{i}' for i in range(L)]
    tn = qtn.TensorNetwork()  # define an empty tensornetwork
    depth = 0
    n_gates = 0
    for n in range(0, steps):
        for j in range(len(a_list)):
            if j % 2 == pattern:
                for i in range(0, L - 1, 2):
                    H_i = Jz * (np.kron(Pz, I) + np.kron(I, Pz))
                    U_i = spla.expm(1j * H_i * dt * a_list[j])
                    ind = (index_list[i], index_list[i + 1], f'G{leg_indx}',
                           f'G{leg_indx + 1}')  # each tensor 'updates' the two bond indices
                    U = qtn.Tensor(U_i.reshape(2, 2, 2, 2), ind,
                                   tags=['A', 'Even', f'SU4_{i}_{i + 1}', f'L{depth}', f'G{n_gates}'])
                    tn = tn & U  # dont do the contraction, yet
                    index_list[i] = f'G{leg_indx}'
                    index_list[i + 1] = f'G{leg_indx + 1}'
                    leg_indx += 2
                    n_gates += 1

                if L % 2:  # If odd, we add one term at the edge.
                    H_i = Jz * np.kron(I, Pz)
                    U_i = spla.expm(1j * H_i * dt * a_list[j])
                    ind = (index_list[L - 2], index_list[L - 1], f'G{leg_indx}',
                           f'G{leg_indx + 1}')  # each tensor 'updates' the two bond indices
                    U = qtn.Tensor(U_i.reshape(2, 2, 2, 2), ind,
                                   tags=['A', 'Odd_edge', f'SU4_{L - 2}_{L - 1}', f'G{n_gates}'])
                    tn = tn & U  # dont do the contraction, yet
                    index_list[L - 2] = f'G{leg_indx}'
                    index_list[L - 1] = f'G{leg_indx + 1}'
                    leg_indx += 2
                    n_gates += 1

            else:
                for i in range(0, L - 1, 2):  # Even
                    H_i = Jxx * np.kron(Px, Px) + Jx * (np.kron(Px, I))
                    if i == L - 2:
                        H_i += Jx * (np.kron(I, Px))
                    U_i = spla.expm(1j * H_i * dt * a_list[j])
                    ind = (index_list[i], index_list[i + 1], f'G{leg_indx}',
                           f'G{leg_indx + 1}')  # each tensor 'updates' the two bond indices
                    U = qtn.Tensor(U_i.reshape(2, 2, 2, 2), ind,
                                   tags=['B', 'Even', f'SU4_{i}_{i + 1}', f'L{depth}', f'G{n_gates}'])
                    tn = tn & U
                    index_list[i] = f'G{leg_indx}'
                    index_list[i + 1] = f'G{leg_indx + 1}'
                    leg_indx += 2
                    n_gates += 1

                for i in range(1, L - 1, 2):  # odd
                    H_i = Jxx * np.kron(Px, Px) + Jx * (np.kron(Px, I))
                    if i == L - 2:
                        H_i += Jx * (np.kron(I, Px))
                    U_i = spla.expm(1j * H_i * dt * a_list[j])
                    ind = (index_list[i], index_list[i + 1], f'G{leg_indx}',
                           f'G{leg_indx + 1}')  # each tensor 'updates' the two bond indices
                    U = qtn.Tensor(U_i.reshape(2, 2, 2, 2), ind,
                                   tags=['B', 'Odd', f'SU4_{i}_{i + 1}', f'L{depth}', f'G{n_gates}'])
                    tn = tn & U
                    index_list[i] = f'G{leg_indx}'
                    index_list[i + 1] = f'G{leg_indx + 1}'
                    leg_indx += 2
                    n_gates += 1

            depth += 1
    for i in range(len(index_list)):
        tn = tn.reindex({index_list[i]: f'b{i}'})
    return tn


def trotter_evolution_optimized_nn_heisenberg_tn(L: int, dt: float, steps: int,
                                                 p: int = 2):  # (B < A) so set B = XX+X, A = Z
    if p == 1:
        a_list = TwoTermsOrder1.a_list
        pattern = TwoTermsOrder1.pattern
    elif p == 2:
        a_list = TwoTermsOrder2.a_list
        pattern = TwoTermsOrder2.pattern
    elif p == 4:
        a_list = TwoTermsOrder4.a_list
        pattern = TwoTermsOrder4.pattern
    elif p == 6:
        a_list = TwoTermsOrder6.a_list
        pattern = TwoTermsOrder6.pattern
    else:
        raise NotImplementedError
    leg_indx = 0
    index_list = [f'k{i}' for i in range(L)]
    tn = qtn.TensorNetwork()  # define an empty tensornetwork
    depth = 0
    n_gates = 0
    for n in range(0, steps):
        for j in range(len(a_list)):
            if j % 2 == pattern:
                for i in range(0, L - 1, 2):  # odd layer
                    H_i = np.kron(Px, Px) + np.kron(Py, Py) + np.kron(Pz, Pz)
                    U_i = spla.expm(1j * H_i * dt * a_list[j])
                    ind = (index_list[i], index_list[i + 1], f'G{leg_indx}',
                           f'G{leg_indx + 1}')  # each tensor 'updates' the two bond indices
                    U = qtn.Tensor(U_i.reshape(2, 2, 2, 2), ind,
                                   tags=['A', f'SU4_{i}_{i + 1}', 'Even', f'L{depth}', f'G{n_gates}'])
                    tn = tn & U  # dont do the contraction, yet
                    index_list[i] = f'G{leg_indx}'
                    index_list[i + 1] = f'G{leg_indx + 1}'
                    leg_indx += 2
                    n_gates += 1

            else:
                for i in range(1, L - 1, 2):  # even layer
                    H_i = np.kron(Px, Px) + np.kron(Py, Py) + np.kron(Pz, Pz)
                    U_i = spla.expm(1j * H_i * dt * a_list[j])
                    ind = (index_list[i], index_list[i + 1], f'G{leg_indx}',
                           f'G{leg_indx + 1}')  # each tensor 'updates' the two bond indices
                    U = qtn.Tensor(U_i.reshape(2, 2, 2, 2), ind,
                                   tags=['B', f'SU4_{i}_{i + 1}', 'Odd', f'L{depth}', f'G{n_gates}'])
                    tn = tn & U
                    index_list[i] = f'G{leg_indx}'
                    index_list[i + 1] = f'G{leg_indx + 1}'
                    leg_indx += 2
                    n_gates += 1

            depth += 1

    for i in range(len(index_list)):
        tn = tn.reindex({index_list[i]: f'b{i}'})
    return tn


def trotter_evolution_optimized_nn_mbl_tn(L: int, delta: float, dh: np.ndarray, dt: float, steps: int,
                                          p: int = 2):  # (B < A) so set B = XX+X, A = Z
    if p == 1:
        a_list = TwoTermsOrder1.a_list
        pattern = TwoTermsOrder1.pattern
    elif p == 2:
        a_list = TwoTermsOrder2.a_list
        pattern = TwoTermsOrder2.pattern
    elif p == 4:
        a_list = TwoTermsOrder4.a_list
        pattern = TwoTermsOrder4.pattern
    elif p == 6:
        a_list = TwoTermsOrder6.a_list
        pattern = TwoTermsOrder6.pattern
    else:
        raise NotImplementedError

    leg_indx = 0
    index_list = [f'k{i}' for i in range(L)]
    tn = qtn.TensorNetwork()  # define an empty tensornetwork
    depth = 0
    n_gates = 0
    for n in range(0, steps):
        for j in range(len(a_list)):
            if j % 2 == pattern:
                for i in range(0, L - 1, 2):  # even layer
                    H_i = (np.kron(Px, Px) + np.kron(Py, Py) + delta * np.kron(Pz, Pz)) + dh[i] * np.kron(Pz, I) + dh[
                        i + 1] * np.kron(I, Pz)
                    U_i = spla.expm(1j * H_i * dt * a_list[j])
                    ind = (index_list[i], index_list[i + 1], f'G{leg_indx}',
                           f'G{leg_indx + 1}')  # each tensor 'updates' the two bond indices
                    U = qtn.Tensor(U_i.reshape(2, 2, 2, 2), ind,
                                   tags=['A', f'SU4_{i}_{i + 1}', 'Even', f'L{depth}', f'G{n_gates}'])
                    tn = tn & U  # dont do the contraction, yet
                    index_list[i] = f'G{leg_indx}'
                    index_list[i + 1] = f'G{leg_indx + 1}'
                    leg_indx += 2
                    n_gates += 1
            else:
                for i in range(1, L - 1, 2):  # odd layer
                    if (i == (L - 2)) and L % 2:  # if odd, add pz to last gate
                        H_i = (np.kron(Px, Px) + np.kron(Py, Py) + delta * np.kron(Pz, Pz)) + dh[-1] * np.kron(I, Pz)
                    else:
                        H_i = (np.kron(Px, Px) + np.kron(Py, Py) + delta * np.kron(Pz, Pz))
                    U_i = spla.expm(1j * H_i * dt * a_list[j])
                    ind = (index_list[i], index_list[i + 1], f'G{leg_indx}',
                           f'G{leg_indx + 1}')  # each tensor 'updates' the two bond indices
                    U = qtn.Tensor(U_i.reshape(2, 2, 2, 2), ind,
                                   tags=['B', f'SU4_{i}_{i + 1}', 'Odd', f'L{depth}', f'G{n_gates}'])
                    tn = tn & U
                    index_list[i] = f'G{leg_indx}'
                    index_list[i + 1] = f'G{leg_indx + 1}'
                    leg_indx += 2
                    n_gates += 1
            depth += 1
    for i in range(len(index_list)):
        tn = tn.reindex({index_list[i]: f'b{i}'})
    return tn


def trotter_evolution_optimized_ising_nnn_tn(L: int, J: float, V: float, dt: float, steps: int,
                                             p: int = 2):  # (B < A) so set B = XX+X, A = Z
    """−J(Xi X_i+1 + Z_i) + V (Z_i Z_i+1 + X_i−1 X_i+1)"""
    if p == 1:
        a_list = TwoTermsOrder1.a_list
        pattern = TwoTermsOrder1.pattern
    elif p == 2:
        a_list = TwoTermsOrder2.a_list
        pattern = TwoTermsOrder2.pattern
    elif p == 4:
        a_list = TwoTermsOrder4.a_list
        pattern = TwoTermsOrder4.pattern
    elif p == 6:
        a_list = TwoTermsOrder6.a_list
        pattern = TwoTermsOrder6.pattern
    else:
        raise NotImplementedError

    leg_indx = 1
    index_list = [f'k{i}' for i in range(L)]
    tn = qtn.TensorNetwork()  # define an empty tensornetwork
    depth = 0
    n_gates = 0
    for n in range(0, steps):
        for j in range(len(a_list)):
            if j % 2 == pattern:
                for i in range(1, L - 1, 1):
                    if i == 1:  # add edge
                        H_i = J * (np.kron(Px, np.kron(Px, I)) + np.kron(I, np.kron(Px, Px))) + \
                              V * np.kron(Px, np.kron(I, Px))
                    else:
                        H_i = J * np.kron(I, np.kron(Px, Px)) + V * np.kron(Px, np.kron(I, Px))
                    U_i = spla.expm(1j * H_i * dt * a_list[j])
                    ind = (index_list[i - 1], index_list[i], index_list[i + 1], f'G{leg_indx - 1}',
                           f'G{leg_indx}', f'G{leg_indx + 1}')  # each tensor 'updates' the two bond indices
                    U = qtn.Tensor(U_i.reshape([2] * 6), ind,
                                   tags=['A', 'Even', f'SU4_{i}_{i + 1}', f'L{depth}', f'G{n_gates}'])
                    tn = tn & U  # dont do the contraction, yet
                    index_list[i - 1] = f'G{leg_indx - 1}'
                    index_list[i] = f'G{leg_indx}'
                    index_list[i + 1] = f'G{leg_indx + 1}'
                    leg_indx += 3
                    n_gates += 1
            else:
                for i in range(1, L - 1, 1):
                    if i == 1:  # add edge
                        H_i = J * (np.kron(I, np.kron(Pz, I)) + np.kron(Pz, np.kron(I, I))) + \
                              V * (np.kron(Pz, np.kron(Pz, I)) + np.kron(I, np.kron(Pz, Pz)))
                    elif i == (L - 2):
                        H_i = J * (np.kron(I, np.kron(Pz, I)) + np.kron(I, np.kron(I, Pz))) + \
                              V * np.kron(I, np.kron(Pz, Pz))
                    else:
                        H_i = J * (np.kron(I, np.kron(Pz, I)) + V * np.kron(I, np.kron(Pz, Pz)))
                    U_i = spla.expm(1j * H_i * dt * a_list[j])
                    ind = (index_list[i - 1], index_list[i], index_list[i + 1], f'G{leg_indx - 1}',
                           f'G{leg_indx}', f'G{leg_indx + 1}')  # each tensor 'updates' the two bond indices
                    U = qtn.Tensor(U_i.reshape([2] * 6), ind,
                                   tags=['B', 'Odd', f'SU4_{i}_{i + 1}', f'L{depth}', f'G{n_gates}'])
                    tn = tn & U  # dont do the contraction, yet
                    index_list[i - 1] = f'G{leg_indx - 1}'
                    index_list[i] = f'G{leg_indx}'
                    index_list[i + 1] = f'G{leg_indx + 1}'
                    leg_indx += 3
                    n_gates += 1

            depth += 1
    for i in range(len(index_list)):
        tn = tn.reindex({index_list[i]: f'b{i}'})
    return tn


def compress_trotterization_into_circuit(L, tn: qtn.TensorNetwork):
    """Compress a trotterized circuit by absorbing gates acting on the same qubits"""
    layers = {}
    layer = 0
    even_or_odd = 'Even' if 'Even' in tn.tensors[0].tags else 'Odd'  # second tag must indicate even or odd.
    layers[layer] = [(tn.tensors[0].copy(), even_or_odd)]
    assert all(
        'Even' in t.tags or 'Odd' in t.tags for t in tn.tensors), 'All tensors must be tagged with `Even` or `Odd`'
    assert all(any('U' in tag for tag in t.tags) for t in tn.tensors), 'All tensors must have a U_i_j tag'
    for tensor in tn.tensors[1:]:
        if even_or_odd in tensor.tags:  # If we are also Even (Odd) add it to the layer
            layers[layer].append((tensor, even_or_odd))
        else:
            layer += 1
            if even_or_odd == 'Even':
                even_or_odd = 'Odd'
            else:
                even_or_odd = 'Even'
            layers[layer] = [(tensor.copy(), even_or_odd)]
    new_tn = qtn.TensorNetwork()
    n_gates = 0
    for l_i in range(layer + 1):
        even_or_odd = layers[l_i][0][1]
        if even_or_odd == 'Even':
            qubit_iterator = list(i for i in range(0, L - 1, 2))
        else:
            qubit_iterator = list(i for i in range(1, L - 1, 2))
        for i in qubit_iterator:
            #TODO: Does not work for Ising NNN.
            gates = list(filter(lambda x: any(f'SU4_{i}_{i + 1}' in x_i for x_i in x[0].tags), layers[l_i]))
            new_gate = np.eye(4, dtype=complex)
            for (g, _) in gates:
                new_gate = new_gate @ g.data.reshape(4, 4)
            # Remove Trotter tag
            tags = gates[0][0].tags
            if "A" in tags:
                tags.remove("A")
            if "B" in tags:
                tags.remove("B")
            # Remove the old layer tag and add new layer tag
            layer_tag = list(filter(lambda x: re.match(r'^L\d+$', x), tags))[0]
            tags.remove(layer_tag)
            tags.add(f'L{l_i}')
            # Remove the old gate tag and add new gate tag
            gate_tag = list(filter(lambda x: re.match(r'^G\d+$', x), tags))[0]
            tags.remove(gate_tag)
            tags.add(f"G{n_gates}")
            # Get the old indices
            left_indices = list(gates[0][0].inds[:2])
            right_indices = list(gates[-1][0].inds[2:])
            # Construct new tensor
            new_gate = qtn.Tensor(data=new_gate.reshape([2] * 4), inds=tuple(left_indices + right_indices),
                                  left_inds=left_indices, tags=tags)
            new_tn.add(new_gate)
            n_gates += 1
    assert_bricklayer(L, new_tn)
    return new_tn


def assert_bricklayer(L: int, circuit: qtn.TensorNetwork):
    qubit_iterator = {'Even': set(i for i in range(0, L, 1)),
                      'Odd': set(i for i in range(1, L - 1, 1))}
    indices_found = set()
    current_even_or_odd = 'Even' if 'Even' in circuit.tensors[0].tags else 'Odd'
    for block in circuit:
        even_or_odd = 'Even' if 'Even' in block.tags else 'Odd'
        # When we switch, make sure all indices have occurred
        if current_even_or_odd != even_or_odd:
            assert qubit_iterator[current_even_or_odd] == indices_found, f'Found only gates on qubits {indices_found}, ' \
                                                                         f'expected gates on {qubit_iterator[current_even_or_odd]} '
            indices_found.clear()
            current_even_or_odd = even_or_odd

        qubits = list(x.split('_')[1:] for x in block.tags if 'SU4_' in x)[0]
        q1, q2 = int(qubits[0]), int(qubits[1])
        indices_found = indices_found.union(set([q1, q2]))
