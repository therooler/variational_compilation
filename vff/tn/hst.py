import quimb as qu
import quimb.tensor as qtn
import numpy as np
import scipy

from .trotter import TwoTermsOrder2, Px, Py, Pz
from .mps_circuit import apply_circuit_to_state


def tensor_to_gate(tensor):
    data = tensor.data.reshape(4, 4)
    for tag in tensor.tags:
        if tag[:4] == 'SU4_':
            sites_str = tag[4:].split('_')
    site1_str = sites_str[0]
    site2_str = sites_str[1]
    site1 = int(site1_str)
    site2 = int(site2_str)
    return data, site1, site2


def hst(L, U, V, cutoff=1e-09, contract='swap+split', trans_inv=True):
    # XXZ model
    psi_U = qtn.MPS_computational_state('0' * (2 * L))  # duplicate the system
    split_opts = {}
    split_opts['cutoff'] = cutoff
    split_opts['method'] = 'qr'
    H = qu.hadamard()
    CNOT = qu.controlled('not')
    # create bell pairs
    for i in range(L):
        psi_U.gate_(H, (2 * i), contract=contract, **split_opts)
        psi_U.gate_(CNOT, (2 * i, 2 * i + 1), contract=contract, **split_opts)

    # apply U (the 'true' circuit) to the left
    print("Applying U")
    past_tensors = U.copy().tensors
    shift = 0
    for tensor in reversed(past_tensors):
        G, site1, site2 = tensor_to_gate(tensor)
        psi_U.gate_(G, (2 * site1 + shift, 2 * site2 + shift), contract=contract, **split_opts)
    print('Done applying U')
    H = qu.hadamard()
    CNOT = qu.controlled('not')
    psi_V = qtn.MPS_computational_state('0' * (2 * L))  # duplicate the system
    # create bell pairs
    for i in range(L):
        psi_V.gate_(H, (2 * i), contract=contract, **split_opts)
        psi_V.gate_(CNOT, (2 * i, 2 * i + 1), contract=contract, **split_opts)
    # Apply V to the right
    print("Applying V")
    past_tensors = V.copy().tensors
    shift = 1
    split_opts['method'] = 'svd'
    for tensor in past_tensors:
        G, site1, site2 = tensor_to_gate(tensor)
        psi_V.gate_(G.T.conj(), (2 * site1 + shift, 2 * site2 + shift), contract=contract, **split_opts)
    print('Done applying V')
    return abs((psi_V & psi_U.H).contract()) ** 2
