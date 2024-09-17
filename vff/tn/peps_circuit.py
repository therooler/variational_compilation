import quimb.tensor as qtn
import quimb as qu

import torch
from typing import List
import re
import numpy as np

from quimb.tensor.decomp import _ISOMETRIZE_METHODS
from autoray import (
    backend_like,
    compose,
    do,
    reshape,
)


@compose
def isometrize_qr_fixed(x, backend=None):
    """Perform isometrization using the QR decomposition. FIX FOR NEW PYTORCH"""
    with backend_like(backend):
        Q, R = do("linalg.qr", x)
        # stabilize qr by fixing diagonal of R in canonical, positive form (we
        # don't actaully do anything to R, just absorb the necessary sign -> Q)
        rd = do("diag", R)
        s = do("sgn", rd) + (rd == 0)
        Q = Q * reshape(s, (1, -1))
        return Q


_ISOMETRIZE_METHODS["qr"] = isometrize_qr_fixed


# useful funcitons
def get_loss_fn(psi_tar, contract_opts):
    def loss_function(psi: qtn.TensorNetwork):
        # compute the total energy, here quimb handles constructing
        # and contracting all the appropriate lightcones
        psi = psi.isometrize(method='qr')
        # psi.balance_bonds_()
        # psi.equalize_norms_(1.0)
        # overlaps = [1 - abs((psi_tar[i] & psi).contract(optimize='auto-hq', **contract_opts)) ** 2 for i in range(len(psi_tar))]
        overlaps = [1 - abs((psi_tar[i] & psi).contract(optimize='auto-hq')) ** 2 for i in range(len(psi_tar))]
        # overlaps = [1 - abs((psi_tar[i] & psi).contract_boundary(**contract_opts)) ** 2 for i in range(len(psi_tar))]
        return sum(overlaps) / len(overlaps)

    return loss_function


class TNModel(torch.nn.Module):

    def __init__(self, psi, psi_tars, device, contract_opts=None):
        super().__init__()
        if contract_opts is None:
            contract_opts = {}
        psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128, device=device))
        for i in range(len(psi_tars)):
            psi_tars[i].apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128, device=device))

        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(psi)
        # n.b. you might want to do extra processing here to e.g. store each
        # parameter as a reshaped matrix (from left_inds -> right_inds), for
        # some optimizers, and for some torch parametrizations
        self.torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })

        self._loss_fn = get_loss_fn(psi_tars, contract_opts)

    def forward(self):
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # isometrize and then return the energy
        return self._loss_fn(psi)


def create_target(L, psi_pqc, psi0, psit):
    '''
    base is assumed to be some random product state
    '''
    psi_tar = psit.H
    for i in range(L):
        psi_tar = psi_tar & qtn.Tensor(psi0.tensors[i].data, psi_pqc.tensors[i].inds)
    return psi_tar


def create_targets(L: int, pqc: qtn.TensorNetwork, psi0: List[qtn.MatrixProductState],
                   psit: List[qtn.MatrixProductState]):
    psi = pqc.tensors[L]
    for i in range(L + 1, len(pqc.tensors)):
        psi = psi & pqc.tensors[i]

    psi_tars = [create_target(L, pqc, psi0[i], psit[i]) for i in range(len(psi0))]

    return psi, psi_tars


def qmps_brick_2d(Lx: int, Ly: int, in_depth: int = 2, val_iden: float = False, rand: bool = True) -> qtn.TensorNetwork:
    psi = qtn.PEPS.empty(Lx, Ly, bond_dim=2)
    # print(psi)
    for t in psi:
        t.modify(left_inds=[t.inds[-1]], tags=[f"I{t.inds[-1]}", "PEPS"])
    n_apply = 0
    for r in range(in_depth):
        if r % 2:
            for ly in range(Ly):
                for lx in range(0, Lx - 1, 2):
                    if rand:
                        G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                    else:
                        G = qu.identity(4, dtype=complex)
                    psi.gate_(G, ((lx, ly), (lx + 1, ly)),
                              tags={f'SU4_{lx},{ly}_{lx + 1},{ly}', f'G{n_apply}', f'L{r}', 'LR'})
                for lx in range(1, Lx - 1, 2):
                    if rand:
                        G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                    else:
                        G = qu.identity(4, dtype=complex)
                    psi.gate_(G, ((lx, ly), (lx + 1, ly)),
                              tags={f'SU4_{lx},{ly}_{lx + 1},{ly}', f'G{n_apply}', f'L{r}', 'LR'})
                    n_apply += 1
        else:
            for lx in range(Lx):
                for ly in range(0, Ly - 1, 2):  # odd
                    if rand:
                        G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                    else:
                        G = qu.identity(4, dtype=complex)
                    psi.gate_(G, ((lx, ly), (lx, ly + 1)),
                              tags={f'SU4_{lx},{ly}_{lx},{ly + 1}', f'G{n_apply}', f'L{r}', 'UD'})
                    n_apply += 1
                for ly in range(1, Ly - 1, 2):  # even
                    if rand:
                        G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                    else:
                        G = qu.identity(4, dtype=complex)
                    psi.gate_(G, ((lx, ly), (lx, ly + 1)),
                              tags={f'SU4_{lx},{ly}_{lx},{ly + 1}', f'G{n_apply}', f'L{r}', 'UD'})
                    n_apply += 1
    return psi.astype_('complex128')  # , list_u3


def load_gates(tn: qtn.TensorNetwork, target_tn: qtn.TensorNetwork, transpose: bool = False):
    """Load gates from one tensor network to the other"""
    gates_tn = set(g for g in tn.tags if bool(re.match(r'^G\d+$', g)))
    gates_target_tn = set(g for g in target_tn.tags if bool(re.match(r'^G\d+$', g)))
    assert gates_tn == gates_target_tn, f"Expected the number of gates in the circuit to be equal," \
                                        f" but found: {gates_tn} and {gates_target_tn}"
    for g in gates_tn:
        ts_tn = tn[g]
        ts_target_tn = target_tn[g]
        qubit_tag_tn = list(filter(lambda x: f'SU4' in x, ts_tn.tags))[0]
        qubit_tag_target_tn = list(filter(lambda x: f'SU4' in x, ts_tn.tags))[0]
        assert qubit_tag_tn == qubit_tag_target_tn, f"Expected gate {g} to act on same qubit, " \
                                                    f"but found tags {qubit_tag_tn} and {qubit_tag_target_tn}"
        if transpose:
            tn[g].modify(data=np.transpose(ts_target_tn.data, axes=(2, 3, 0, 1)), left_inds=ts_tn.left_inds)
        else:
            tn[g].modify(data=ts_target_tn.data, left_inds=ts_tn.left_inds)
