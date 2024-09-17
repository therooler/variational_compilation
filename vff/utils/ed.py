import scipy.sparse
import numpy as np

paulis = {'I': scipy.sparse.csr_matrix(np.eye(2).astype(np.complex64)),
          'x': scipy.sparse.csr_matrix(np.array([[0, 1], [1, 0]]).astype(np.complex64)),
          'y': scipy.sparse.csr_matrix(np.array([[0, -1j], [1j, 0]]).astype(np.complex64)),
          'z': scipy.sparse.csr_matrix(np.array([[1, 0], [0, -1]]).astype(np.complex64))}


def build_two_body(interactions, L: int) -> scipy.sparse.csr_matrix:
    ham = scipy.sparse.csr_matrix((int(2 ** L), (int(2 ** L))), dtype=complex)
    for (scalar, term_1, term_2) in interactions:
        # XX term
        tprod = ["I" for _ in range(L)]
        loc_i = term_1[1]
        loc_j = term_2[1]
        tprod[loc_i] = term_1[0]
        tprod[loc_j] = term_2[0]
        p = paulis[tprod[0]]
        for op in range(1, L):
            p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
        if callable(scalar):
            ham += scalar(loc_i) * p
        else:
            ham += scalar * p
    return ham


def build_one_body(interactions, L: int) -> scipy.sparse.csr_matrix:
    ham = scipy.sparse.csr_matrix((int(2 ** L), (int(2 ** L))), dtype=complex)
    for (scalar, term_1) in interactions:
        # XX term
        tprod = ["I" for _ in range(L)]
        loc_i = term_1[1]
        tprod[loc_i] = term_1[0]
        p = paulis[tprod[0]]
        for op in range(1, L):
            p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
        if callable(scalar):
            ham += scalar(loc_i) * p
        else:
            ham += scalar * p
    return ham


def build_XXZ_matrix(L: int, J: float, Delta: float, pbc: bool = False) -> scipy.sparse.csr_matrix:
    """
    builds tfim Hamiltonian
    """
    ## Setup basis

    ## Operator lists
    if pbc:
        J_xx = [[J, ('x', i), ('x', (i + 1) % L)] for i in range(L)]  # PBC
        J_yy = [[J, ('y', i), ('y', (i + 1) % L)] for i in range(L)]  # PBC
        J_zz = [[Delta * J, ('z', i), ('z', (i + 1) % L)] for i in range(L)]  # PBC
    else:
        J_xx = [[J, ('x', i), ('x', (i + 1) % L)] for i in range(L - 1)]  # OBC
        J_yy = [[J, ('y', i), ('y', (i + 1) % L)] for i in range(L - 1)]  # OBC
        J_zz = [[Delta * J, ('z', i), ('z', (i + 1) % L)] for i in range(L - 1)]  # OBC
    interactions = J_xx + J_yy + J_zz
    H = build_two_body(interactions, L)
    return H


def build_mbl_matrix(L: int, delta, dh: np.ndarray, pbc: bool = False) -> scipy.sparse.csr_matrix:
    """
    builds tfim Hamiltonian
    """
    ## Setup basis

    ## Operator lists
    if pbc:
        J_xx = [[1, ('x', i), ('x', (i + 1) % L)] for i in range(L)]  # PBC
        J_yy = [[1, ('y', i), ('y', (i + 1) % L)] for i in range(L)]  # PBC
        J_zz = [[delta, ('z', i), ('z', (i + 1) % L)] for i in range(L)]  # PBC
        h_z = [[lambda i: dh[i], ('z', i)] for i in range(L)]  # PBC
    else:
        J_xx = [[1, ('x', i), ('x', (i + 1) % L)] for i in range(L - 1)]  # OBC
        J_yy = [[1, ('y', i), ('y', (i + 1) % L)] for i in range(L - 1)]  # OBC
        J_zz = [[delta, ('z', i), ('z', (i + 1) % L)] for i in range(L - 1)]  # OBC
        h_z = [[lambda i: dh[i], ('z', i)] for i in range(L)]  # OBC

    interactions = J_xx + J_yy + J_zz
    H2 = build_two_body(interactions, L)
    H1 = build_one_body(h_z, L)
    return H2 + H1


def build_SDIsing_matrix(L: int, J: float, V: float, pbc: bool = False) -> scipy.sparse.csr_matrix:
    """
    builds tfim Hamiltonian with next nearest neighbors interactions
    $$H_{\rm SDIM} = \Sigma_{i} -J(X_i X_{i+1} + Z_{i})+V(Z_i Z_{i+1} + X_{i-1} X_{i+1})$$
    """
    ## Operator lists
    if pbc:
        J_xx = [[J, ('x', i), ('x', (i + 1) % L)] for i in range(L)]  # PBC
        J_z = [[J, ('z', i)] for i in range(L)]  # OBC
        V_xx = [[V, ('x', i - 1), ('x', (i + 1) % L)] for i in range(1, L, 1)]  # PBC
        V_zz = [[V, ('z', i), ('z', (i + 1) % L)] for i in range(L)]  # PBC
    else:
        J_xx = [[J, ('x', i), ('x', (i + 1) % L)] for i in range(L - 1)]  # OBC
        J_z = [[J, ('z', i)] for i in range(L)]  # OBC
        V_xx = [[V, ('x', i - 1), ('x', (i + 1) % L)] for i in range(1, L - 1, 1)]  # OBC
        V_zz = [[V, ('z', i), ('z', (i + 1) % L)] for i in range(L - 1)]  # OBC

    interactions_two_body = J_xx + V_xx + V_zz
    ham_two = build_two_body(interactions_two_body, L)
    ham_one = build_one_body(J_z, L)
    ham = ham_two + ham_one
    return ham


# here are the ED results

def build_TFIZ_matrix(L: int, Jxx: float, Jz: float, Jx: float, pbc: bool = False) -> scipy.sparse.csr_matrix:
    """
    builds tfimz Hamiltonian
    """

    ## Operator lists
    if pbc:
        J_xx = [[Jxx, ('x', i), ('x', (i + 1) % L)] for i in range(L)]  # PBC
        J_x = [[Jx, ('x', i)] for i in range(L)]  # PBC
        J_z = [[Jz, ('z', i)] for i in range(L)]  # PBC
    else:
        J_xx = [[Jxx, ('x', i), ('x', (i + 1) % L)] for i in range(L - 1)]  # OBC
        J_x = [[Jx, ('x', i)] for i in range(L)]  # PBC
        J_z = [[Jz, ('z', i)] for i in range(L)]  # PBC

    ham_two = build_two_body(J_xx, L)
    interactions_one_body = J_x + J_z
    ham_one = build_one_body(interactions_one_body, L)
    ham = ham_two + ham_one

    return ham


def build_heisenberg_matrix(L, pbc=False) -> scipy.sparse.csr_matrix:
    """
    builds tfimz Hamiltonian
    """

    ## Operator lists

    return build_XXZ_matrix(L, 1.0, 1.0, pbc)


def build_TFIM_matrix(L, Jxx: float, Jz: float, pbc=False):
    """
    builds tfimz Hamiltonian
    """

    ## Operator lists

    return build_TFIZ_matrix(L, Jxx, Jz, 0.0, pbc)
