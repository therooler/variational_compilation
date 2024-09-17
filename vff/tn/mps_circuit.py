import quimb.tensor as qtn
import quimb as qu
import cotengra as ctg
import torch
import numpy as np
import warnings
from typing import List
import re
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

opti = ctg.ReusableHyperOptimizer(
    progbar=True,
    minimize='combo',
    methods=['spinglass'],#'spinglass','greedy','labelprop',
    reconf_opts={},
    max_repeats=64,
    optlib='random',
    directory=True  # set this for persistent cache
)


_ISOMETRIZE_METHODS["qr"] = isometrize_qr_fixed

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    import cotengra as ctg


def create_target(L: int, psi_pqc: qtn.MatrixProductState, psi0: qtn.TensorNetwork, psit: qtn.TensorNetwork):
    """
    base is assumed to be some random product state

    Args:
        L:
        psi_pqc:
        psi0:
        psit:

    Returns:

    """
    psi_tar = psit.H
    psi_tar.add_tag('psit')
    for i in range(L):
        psi_tar = psi_tar & qtn.Tensor(psi0.tensors[i].data, psi_pqc.tensors[i].inds, tags=('psi0',))
    return psi_tar


def create_target_single_tn(L: int, psi_pqc: qtn.MatrixProductState, psi0: List[qtn.MatrixProductState],
                            psit: List[qtn.MatrixProductState]):
    """
    base is assumed to be some random product state
    want to get rid of the loop,
    and store the thing as one big tensor network
    """

    r = len(psi0)
    psi0_tensors = []
    psit_tensors = []
    ## generate the data for each tensor
    for j in range(len(psi0[0].tensors)):
        tensor_psi0 = np.zeros(r * np.product(psi0[0].tensors[j].shape)).reshape(
            [r] + list(psi0[0].tensors[j].shape)) + 0 * 1j
        for i in range(r):
            tensor_psi0[i] = psi0[i].tensors[j].data
        ts = qtn.Tensor(tensor_psi0,
                        tuple(['r'] + list(psi0[0].tensors[j].inds)[:-1] + [psi_pqc.tensors[j].inds[-1]]))

        psi0_tensors.append(ts)

    for j in range(len(psit[0].tensors)):
        tensor_psit = np.zeros(r * np.product(psit[0].tensors[j].shape)).reshape(
            [r] + list(psit[0].tensors[j].shape)) + 0 * 1j
        for i in range(r):
            tensor_psit[i] = psit[i].H.tensors[j].data
        ts = qtn.Tensor(tensor_psit, tuple(['r'] + list(psit[0].tensors[j].inds)))
        psit_tensors.append(ts)

    tn = psi0_tensors[0]
    for i in range(1, len(psi0_tensors)):
        tn = tn & psi0_tensors[i]
    for i in range(len(psit_tensors)):
        tn = tn & psit_tensors[i]
    return tn


def norm_fn(psi: qtn.TensorNetwork):
    # parametrize our tensors as isometric/unitary
    return psi.isometrize(method='qr')


def get_loss_fn(psi_tar: List[qtn.MatrixProductState], contract_opts, ctg=False):

    def loss_fn(psi: qtn.TensorNetwork):
        # compute the total energy, here quimb handles constructing
        # and contracting all the appropriate lightcones
        sample_size = len(psi_tar)
        if ctg:
            overlaps = [abs((psi_tar[i] & psi).contract(optimize=opti, **contract_opts)) ** 2 for i in range(sample_size)]
        else:
            overlaps = [abs((psi_tar[i] & psi).contract(**contract_opts)) ** 2 for i in range(sample_size)]
        return 1 - sum(overlaps) / sample_size

    return loss_fn


class TNModel(torch.nn.Module):

    def __init__(self, tn: qtn.TensorNetwork, psi_tar: List[qtn.MatrixProductState] = None, translation: bool = False,
                 contract_opts: dict = None, ctg=False):
        super().__init__()
        if contract_opts is None:
            contract_opts = {}
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(tn)
        # n.b. you might want to do extra processing here to e.g. store each
        # parameter as a reshaped matrix (from left_inds -> right_inds), for
        # some optimizers, and for some torch parametrizations
        self.translation = translation
        self.number_of_gates = len(params)
        if self.translation:
            translation_parameters = {}
            for i in range(self.number_of_gates):
                layer_tag = next(filter(lambda x: re.match(r'^L\d+$', x), self.skeleton.tensors[i].tags))
                layer_i = layer_tag[1:]
                if layer_i not in translation_parameters.keys():
                    translation_parameters[layer_i] = torch.nn.Parameter(params[i])
            self.torch_params = torch.nn.ParameterDict(translation_parameters)
        else:
            self.torch_params = torch.nn.ParameterDict({
                # torch requires strings as keys
                str(i): torch.nn.Parameter(initial)
                for i, initial in params.items()
            })

        if psi_tar is not None:
            loss_fn = get_loss_fn(psi_tar, contract_opts, ctg=ctg)
            self.loss_fn = lambda x: loss_fn(norm_fn(x))

    def forward(self):
        # convert back to original int key format
        if self.translation:
            params = {}
            for i in range(self.number_of_gates):
                layer_tag = next(filter(lambda x: re.match(r'^L\d+$', x), self.skeleton.tensors[i].tags))
                layer_i = layer_tag[1:]
                params[i] = self.torch_params[layer_i]
        else:
            params = {int(i): p for i, p in list(self.torch_params.items())}
        # reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # isometrize and then return the energy
        return self.loss_fn(psi)


def _range_unitary(psi: qtn.MatrixProductState, depth: int, n_Qbit: int, Qubit_ara: int, val_iden: float = 0.,
                   rand: bool = False, start_even: bool = True):
    assert n_Qbit > 0, "Can only make a brick layer circuit for n_Qubit>=1"
    n_apply = 0
    c_val = 0
    for r in range(depth):

        if ((r + start_even) % 2):
            for i in range(0, n_Qbit, 2):
                # print("U_e", i, i + 1, n_apply)
                if rand:
                    G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                else:
                    G = qu.identity(4, dtype=complex)
                psi.gate_(G, (i, i + 1),
                          tags={f'SU4_{i}_{i + 1}', f'G{n_apply}', f'L{r}', 'Even'})
                n_apply += 1
                c_val += 1

        else:
            for i in range(0, n_Qbit - 1, 2):
                # print("U_o", i+1, i + 2, n_apply)
                if rand:
                    G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                else:
                    G = qu.identity(4, dtype=complex)
                psi.gate_(G, (i + 1, i + 2),
                          tags={f'SU4_{i + 1}_{i + 2}', f'G{n_apply}', f'L{r}', 'Odd'})
                n_apply += 1
                c_val += 1

    return n_apply


def qmps_brick(L: int, in_depth: int = 2, val_iden: float = False, rand: bool = True,
               start_even=True) -> qtn.TensorNetwork:
    psi = qtn.MPS_computational_state('0' * L)

    for i in range(L):
        t = psi[i]
        indx = 'k' + str(i)
        t.modify(left_inds=[indx])

    for t in range(L):
        psi[t].modify(tags=[f"I{t}", "MPS"])

    _range_unitary(psi, in_depth, L - 1, L - 1, val_iden=val_iden, rand=rand, start_even=start_even)

    return psi.astype_('complex128')  # , list_u3


def create_targets(L: int, pqc: qtn.TensorNetwork, psi0: List[qtn.MatrixProductState],
                   psit: List[qtn.MatrixProductState], device=None):
    psi = pqc.tensors[L]
    for i in range(L + 1, len(pqc.tensors)):
        psi = psi & pqc.tensors[i]
    psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128, device=device))

    psi_tars = [create_target(L, pqc, psi0[i], psit[i]) for i in range(len(psi0))]

    for i in range(len(psi_tars)):
        psi_tars[i].apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128, device=device))
    return psi, psi_tars


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


def apply_circuit_to_state(L: int, model, psi0, tebd_opts, translation=False):
    if isinstance(model, torch.nn.Module):
        circuit = model.skeleton
    else:
        circuit = model
    if isinstance(psi0, (list, tuple)):
        psit_var = [p.copy() for p in psi0]
    else:
        psit_var = [psi0.copy(), ]
    for psi in psit_var:
        for i in range(L):
            tn = psi[i]
            tn.modify(left_inds=[f"k{i}"])
    for gate in circuit:
        if translation:  # Get the layer tag
            idx = list(filter(lambda x: re.match(r'^L\d+$', x), gate.tags))[0]
        else:  # Get the Gate tag
            idx = list(filter(lambda x: re.match(r'^G\d+$', x), gate.tags))[0]
        if isinstance(model, torch.nn.Module):
            gate_array = model.torch_params[f"{idx[1:]}"].detach().numpy()
        else:
            gate_array = gate.data
        gate.modify(data=gate_array, left_inds=gate.left_inds)

    circuit.isometrize(method='qr', inplace=True)
    for gate in circuit:
        qubits = list(filter(lambda x: 'SU4_' in x, gate.tags))[0]
        qubits = qubits.split('_')[1:]
        q1, q2 = int(qubits[0]), int(qubits[1])
        gate_tn = qtn.Tensor(np.transpose(gate.data, axes=(2, 3, 0, 1)), inds=gate.inds,
                             left_inds=gate.left_inds)
        for psi in psit_var:
            psi.gate_(gate_tn.data, (q1, q2,), contract='swap+split', inplace=True, **tebd_opts)

    if isinstance(psi0, (list, tuple)):
        return psit_var
    else:
        return psit_var[0]


def apply_2d_circuit_to_state(Lx: int, Ly: int, model, psi0, tebd_opts, translation=False):
    L = Lx * Ly
    if isinstance(model, torch.nn.Module):
        circuit = model.skeleton
    else:
        circuit = model
    dic_2d_1d = snake_index(Lx, Ly)
    dic_1d_2d = dict(zip(dic_2d_1d.values(), dic_2d_1d.keys()))
    if isinstance(psi0, (list, tuple)):
        psit_var = [p.copy() for p in psi0]
    else:
        psit_var = [psi0.copy(), ]
    for psi in psit_var:
        for i in range(L):
            tn = psi[i]
            tn.reindex({f"k{dic_1d_2d[i]}": f"k{i}"}, inplace=True)
            tn.modify(left_inds=[f"k{i}"])

    for gate in circuit:
        if translation:  # Get the layer tag
            idx = list(filter(lambda x: re.match(r'^L\d+$', x), gate.tags))[0]
        else:  # Get the Gate tag
            idx = list(filter(lambda x: re.match(r'^G\d+$', x), gate.tags))[0]
        gate_array = model.torch_params[f"{idx[1:]}"].detach().numpy()
        gate.modify(data=gate_array, left_inds=gate.left_inds)

    circuit.isometrize(method='qr', inplace=True)
    for gate in circuit:
        qubits = list(filter(lambda x: 'SU4_' in x, gate.tags))[0]
        qubits = qubits.split('_')[1:]
        q1, q2 = eval(qubits[0]), eval(qubits[1])
        gate_tn = qtn.Tensor(np.transpose(gate.data, axes=(2, 3, 0, 1)), inds=gate.inds,
                             left_inds=gate.left_inds)
        for psi in psit_var:
            psi.gate_(gate_tn.data, (dic_2d_1d[q1], dic_2d_1d[q2]), contract='swap+split', inplace=True, **tebd_opts)
    for psi in psit_var:
        for i in range(L):
            tn2 = psi[i]
            tn2.reindex({f"k{i}": f"k{dic_1d_2d[i]}"}, inplace=True)

    if isinstance(psi0, (list, tuple)):
        return psit_var
    else:
        return psit_var[0]


def snake_index(Lx, Ly):
    dic = {}
    idx = 0
    for i in range(Ly):
        if i % 2 == 0:
            for j in range(Lx):
                dic[(j, i)] = idx
                idx += 1
        else:
            for j in range(Lx - 1, -1, -1):
                dic[(j, i)] = idx
                idx += 1
    return dic


def qmps_brick_quasi_1d(Lx: int, Ly: int, in_depth: int = 2, val_iden: float = False,
                        rand: bool = True, boundary_condition=(False, False)) -> qtn.TensorNetwork:
    psi = qtn.MPS_rand_state(Lx * Ly, bond_dim=1)
    dic_2d_1d = snake_index(Lx, Ly)

    n_apply = 0
    for r in range(in_depth):
        if r % 2:
            for ly in range(Ly):
                for lx in range(0, Lx - 1, 2):
                    if rand:
                        G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                    else:
                        G = qu.identity(4, dtype=complex)
                    psi.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx + 1, ly)]),
                              tags={f'SU4_{lx},{ly}_{lx + 1},{ly}', f'G{n_apply}', f'L{r}00', 'LR'})
                    n_apply += 1
                for lx in range(1, Lx - 1, 2):
                    if rand:
                        G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                    else:
                        G = qu.identity(4, dtype=complex)
                    psi.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx + 1, ly)]),
                              tags={f'SU4_{lx},{ly}_{lx + 1},{ly}', f'G{n_apply}', f'L{r}01', 'LR'})
                    n_apply += 1
                if boundary_condition[0]:
                    if rand:
                        G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                    else:
                        G = qu.identity(4, dtype=complex)
                    psi.gate_(G, (dic_2d_1d[(0, ly)], dic_2d_1d[(Lx - 1, ly)]),
                              tags={f'SU4_{0},{ly}_{Lx - 1},{ly}', f'G{n_apply}', f'L{r}01', 'LR'})
                    n_apply += 1
        else:
            for lx in range(Lx):
                for ly in range(0, Ly - 1, 2):  # odd
                    if rand:
                        G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                    else:
                        G = qu.identity(4, dtype=complex)
                    psi.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx, ly + 1)]),
                              tags={f'SU4_{lx},{ly}_{lx},{ly + 1}', f'G{n_apply}', f'L{r}00', 'UD'})
                    n_apply += 1
                for ly in range(1, Ly - 1, 2):  # even
                    if rand:
                        G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                    else:
                        G = qu.identity(4, dtype=complex)
                    psi.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx, ly + 1)]),
                              tags={f'SU4_{lx},{ly}_{lx},{ly + 1}', f'G{n_apply}', f'L{r}01', 'UD'})
                    n_apply += 1
                if boundary_condition[1]:
                    if rand:
                        G = qu.rand_uni(4, dtype=complex) + qu.rand_uni(4, dtype=complex) * val_iden
                    else:
                        G = qu.identity(4, dtype=complex)
                    psi.gate_(G, (dic_2d_1d[(lx, 0)], dic_2d_1d[(lx, Ly - 1)]),
                              tags={f'SU4_{lx},{0}_{lx},{Ly-1}', f'G{n_apply}', f'L{r}01', 'LR'})
                    n_apply += 1
                    pass
    psi = psi.reindex({'k' + str(v): 'k' + str(k) for k, v in dic_2d_1d.items()})
    return psi.astype_('complex128')  # , list_u3
