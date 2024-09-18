import random

import quimb.tensor as qtn
import quimb
import numpy as np
import scipy.linalg as spla
import tqdm

from .tebd import apply_tebd
from .tebd_quasi_1d import quasi_1d_tebd_heisenberg
from .tebd_nnn_ising import tebd_ising_nnn


def get_make_data_set_fn(hamiltonian, H, tebd_granularity: dict, tebd_opts: dict, PRINT: bool):
    """Function that prepares the data set creator"""

    def make_data_set(get_training_state_fn, t_fn, num_samples_fn, seed):
        psi0_list = []
        psit_list = []
        tebd_errors_list = []
        if hamiltonian == 'ising_nnn':
            for _seed_i in tqdm.tqdm(range(num_samples_fn), disable=not PRINT):
                psi0 = get_training_state_fn(seed + _seed_i)
                L, J, V = H
                psi0, psit = tebd_ising_nnn(psi0, L, t_fn, J, V, tebd_granularity, cutoff=tebd_opts['cutoff'], p=2)
                psi0_list.append(psi0)
                psit_list.append(psit)
                tebd_errors_list.append(np.nan)
        elif hamiltonian == 'heisenberg_2d':
            for _seed_i in tqdm.tqdm(range(num_samples_fn), disable=not PRINT):
                psi0 = get_training_state_fn(seed + _seed_i)
                Lx, Ly, bc = H
                psi0, psit = quasi_1d_tebd_heisenberg(psi0, Lx, Ly, t_fn, tebd_granularity, cutoff=tebd_opts['cutoff'],
                                                      boundary_condition=bc)
                psi0_list.append(psi0)
                psit_list.append(psit)
                tebd_errors_list.append(np.nan)
        else:
            for _seed_i in tqdm.tqdm(range(num_samples_fn), disable=not PRINT):
                ts = np.linspace(0, t_fn, tebd_granularity)
                psi0 = get_training_state_fn(seed + _seed_i)
                psit, tebd_error = apply_tebd(psi0, H, ts, tebd_opts, PRINT=False)
                psi0_list.append(psi0)
                psit_list.append(psit)
                tebd_errors_list.append(tebd_error)
            # Reset the seed
        return psi0_list, psit_list, tebd_errors_list

    return make_data_set


def random_product_state(L: int, seed: int = None):
    return qtn.MPS_rand_state(L, bond_dim=1, dtype='complex128', seed=seed)


def plus_state(L: int, seed: int = None):
    psi0 = qtn.MPS_computational_state('0' * L, tags='PSI0')

    # apply hadamard to each site
    H = quimb.hadamard()
    for i in range(L):
        psi0.gate_(H, i, tags='H', contract=True)
    return psi0


def random_mps_state(L: int, bond_dim, seed: int = None):
    mps = qtn.MPS_rand_state(L, bond_dim=bond_dim, dtype='complex128', seed=seed)
    mps.compress()
    return mps


def random_sz_conserving_state(L: int, total_charge: int = 0, seed: int = None, shallow_circuit: bool = True):
    charge_string = '1' * total_charge + '0' * (L - total_charge)
    ls = list(charge_string)
    random.seed(seed)
    random.shuffle(ls)
    charge_string_random = ''.join(ls)
    mps = qtn.MPS_computational_state(charge_string_random, dtype='complex128')
    if shallow_circuit == True:
        for i in range(0, L - 1, 1):
            parameters = np.random.random(5) * np.pi
            gate = quimb.fsimg(*parameters)
            mps.gate_(gate, (i, i + 1), contract='swap+split')
    return mps


def random_haar_state(L: int, seed: int = None):
    assert L < 16, 'L is too large for an exact states'
    psi = quimb.rand_haar_state(2 ** L, seed=seed)
    Q = round(np.log(psi.size) / np.log(2))
    a = np.reshape(psi, (1, -1))
    mps = []
    for n in range(Q - 1):
        a = np.reshape(a, (a.shape[0] * 2, -1))
        u, g, v = spla.svd(a, full_matrices=False)
        chi_0 = np.sum(g > -1)
        chi = np.min([chi_0, 2 ** (L // 2)])
        g = g[:chi]
        g = np.diag(g)
        u = u[:, :chi]
        v = v[:chi, :]
        z = np.reshape(u, (a.shape[0] // 2, 2, -1))
        mps.append(np.transpose(z, [0, 2, 1]))
        a = g @ v
    a = np.reshape(a, (1, a.shape[0], 2))
    mps.append(np.transpose(a, [0, 2, 1]))
    mps[0] = mps[0].squeeze()
    mps[-1] = mps[-1].squeeze()
    mps = qtn.MatrixProductState(mps)
    mps.compress()
    return mps


def random_U1_state(L, max_bd, num_particles, seed: int = None):
    np.random.seed(seed)
    mps = construct_U1_MPS(L, max_bd, num_particles, fill='crand')
    mps = full_tensors(mps)
    mps = normalize_MPS(mps)
    # Squeeze the edges
    mps[0] = mps[0].squeeze(axis=0)
    mps[-1] = mps[-1].squeeze(axis=1)
    return qtn.MatrixProductState(mps)


#  rpscon v0.1.2.3 (traces and trivial indices)
#
#  This version by Roeland Wiersema, 2024, adapted from Lukasz' code and an earlier version of the
#  scon function by Guifre Vidal. Incorporates con2t by "Frank" (Verstraete?).
#
#  AA = {A1, A2, ..., Ap} cell of tensors
#  v = {v1, v2, ..., vp} cell of vectors
#  e.g. v1 = [3 4 -1] labels the three indices of tensor A1, with -1 indicating an uncontracted index (open leg)
#  [x 1] tensors receive special treatment
#  ord, if present, contains a list of all indices ordered - if not, [1 2 3 4 ..] by default
#  ford, if present, contains the final ordering of the uncontracted indices - if not, [-1 -2 ..] by default
#  This version: Handles traces on a single tensor
#                Handles trivial indices, including trailing ones (suppressed by Matlab)
#
#  Change list:
#  v0.1.2.3: Added support for networks where a disconnected portion reduces to a number, e.g. scon({A,B},{[1 1],[-1 -2]})
#  v0.1.2.2: Fixed support for scon({A,B},{[-1 -2],[-3 -4]})
#  v0.1.2: Now detects if there are multiple parts left after contracting all positive indices.
#          If so, automatically inserts trivial indices and contracts them.
#  v0.1.1: Fixed bug causing crash when output is a number
#  v0.1.0: Created from Guifre's scon.m.


def construct_U1_MPS(Q, max_bd, num_of_particles=1, fill='crand'):
    A = {}
    bd = [None] * (Q + 1)
    max_bd -= 2
    # U(1) sector bond dimensions
    bd[0] = [1]
    if max_bd > 0:
        bd[1] = [1, 1]
        for n in range(2, Q - 1):
            chi = min(2 ** (n - 1), 2 ** (Q - n - 1), max_bd)
            bd[n] = [1, chi, 1]
        bd[Q - 1] = [1, 1]
    else:
        for n in range(1, Q):
            bd[n] = [1]
    bd[Q] = [1]

    z = np.zeros((Q, 2), dtype=int)

    if num_of_particles <= Q / 2:
        N = num_of_particles
        p = np.array([0] + [round(Q * i / (num_of_particles)) for i in range(N)])
        num_p = np.array(list(set(range(Q)) - set(p)))
        z[num_p - 1, 1] = -1
        z[p - 1, 0] = 1

    else:
        N = Q - num_of_particles
        p = np.array([1] + [round(Q * i / (N - 1)) for i in range(N - 1)])
        num_p = np.array(list(set(range(Q)) - set(p)))
        z[num_p - 1, 0] = -1
        z[p - 1, 1] = 1
    for n in range(Q):
        A[n] = [U1SymMat(bd[n], bd[n + 1], z[n, 0], fill),
                U1SymMat(bd[n], bd[n + 1], z[n, 1], fill)]
    return A


def full_tensors(mps):
    B = []
    for n in range(len(mps)):
        s = mps[n][0].shape
        temp = np.zeros((s[0], s[1], 2), complex)
        for k in range(2):
            temp[:, :, k] = mps[n][k].full()
        B.append(temp)
    return B


def construct_particle_operator(Q):
    N = []
    n = np.array([[0, 0], [0, 1]])
    N.append(np.zeros((1, 2, 2, 2), complex))
    N[0][0, 0, :, :] = n
    N[0][0, 1, :, :] = np.eye(2)

    for k in range(1, Q - 1):
        Nk = np.zeros((2, 2, 2, 2))
        Nk[0, 0, :, :] = np.eye(2)
        Nk[1, 0, :, :] = n
        Nk[1, 1, :, :] = np.eye(2)
        N.append(Nk)

    NQ = np.zeros((2, 1, 2, 2))
    NQ[0, 0, :, :] = np.eye(2)
    NQ[1, 0, :, :] = n
    N.append(NQ)
    return N


def normalize_MPS(mps):
    L = np.array([[1.]])
    for n in range(len(mps)):
        Lnew = np.einsum('imk, jnk -> ijmn', mps[n], np.conj(mps[n]))
        L = np.einsum('ij, ijmn -> mn', L, Lnew)
        no = np.trace(L)
        mps[n] = mps[n] / np.sqrt(no)
        L = L / no
    return mps


def measure_particle_number(mps):
    Q = len(mps)
    N = construct_particle_operator(Q)
    L = np.array([[[1.]]])
    for n in range(Q):
        left_calc = np.einsum('ijk, jnm->iknm', L, mps[n])
        right_calc = np.einsum('abcm, qrm->abcqr', N[n], np.conj(mps[n]))
        L = np.einsum('aknm,abmkr->bnr', left_calc, right_calc)
    return np.real(L)


class U1SymMat:
    def __init__(self, SizeR, SizeC, Charge, Fill):
        self.SizeR = SizeR
        self.SizeC = SizeC
        self.Charge = Charge
        self.nzBlocks = []
        self.Blocks = []
        self._construct_blocks(Fill)

    def _construct_blocks(self, Fill):
        if self.Charge >= 0:
            NnzBlocks = max(0, min(len(self.SizeR), len(self.SizeC) - self.Charge))
        else:
            NnzBlocks = max(0, min(len(self.SizeR) + self.Charge, len(self.SizeC)))

        if self.Charge >= 0:
            self.nzBlocks = np.vstack((np.arange(0, NnzBlocks), np.arange(0, NnzBlocks) + self.Charge)).T
        else:
            self.nzBlocks = np.vstack((np.arange(0, NnzBlocks) - self.Charge, np.arange(0, NnzBlocks))).T

        for n in range(NnzBlocks):
            if Fill == 'rand':
                self.Blocks.append(
                    np.random.rand(self.SizeR[self.nzBlocks[n, 0]], self.SizeC[self.nzBlocks[n, 1]]))
            elif Fill == 'crand':
                self.Blocks.append(
                    np.random.rand(self.SizeR[self.nzBlocks[n, 0]],
                                   self.SizeC[self.nzBlocks[n, 1]]) +
                    1j * np.random.rand(self.SizeR[self.nzBlocks[n, 0]],
                                        self.SizeC[self.nzBlocks[n, 1]]))
            elif Fill == 'eye':
                self.Blocks.append(np.eye(self.SizeR[self.nzBlocks[n, 0]]))
                self.Charge = 0
            elif Fill == 'Z':
                self.Blocks.append((-1) ** (n + 1) * np.eye(self.SizeR[self.nzBlocks[n, 0]]))
                self.Charge = 0
            elif Fill == 'zero':
                self.Blocks.append(
                    np.zeros((self.SizeR[self.nzBlocks[n, 0]], self.SizeC[self.nzBlocks[n, 1]])))
            elif Fill == 'null':
                self.Blocks.append(None)
            elif Fill == 'empty':
                self.nzBlocks = []

    def __str__(self):
        dm = np.empty((len(self.SizeR) + 1, len(self.SizeC) + 1), dtype=object)
        for k in range(1, len(self.SizeR) + 1):
            dm[k, 0] = self.SizeR[k - 1]
        for k in range(1, len(self.SizeC) + 1):
            dm[0, k] = self.SizeC[k - 1]

        for n in range(self.nzBlocks.shape[0]):
            dm[self.nzBlocks[n, 0], self.nzBlocks[n, 1]] = 10

        return str(dm)

    @property
    def shape(self):
        return (sum(self.SizeR), sum(self.SizeC))

    def full(self):
        B = np.zeros((sum(self.SizeR), sum(self.SizeC)), dtype=complex)
        for n in range(self.nzBlocks.shape[0]):
            posR = int(np.sum(self.SizeR[:self.nzBlocks[n, 0]]))
            posC = int(np.sum(self.SizeC[:self.nzBlocks[n, 1]]))

            if self.Blocks[n] is None:
                B[posR:posR + self.SizeR[self.nzBlocks[n, 0]], posC:posC + self.SizeC[self.nzBlocks[n, 1]]] = \
                    np.zeros((self.SizeR[self.nzBlocks[n, 0]], self.SizeC[self.nzBlocks[n, 1]]))
            else:
                B[posR:posR + self.SizeR[self.nzBlocks[n, 0]], posC:posC + self.SizeC[self.nzBlocks[n, 1]]] = \
                    self.Blocks[n]
        return B

    def BlockPos(self, pr, pc):
        ra = np.where(self.nzBlocks[:, 0] == pr)[0]
        ca = np.where(self.nzBlocks[:, 1] == pc)[0]

        if np.array_equal(ra, ca):
            return ra + 1
        else:
            raise ValueError('no such block')

    def AddBlock(self, dir, dim):
        if dir == 'c':
            B = U1SymMat(self.SizeR, [dim] + self.SizeC + [dim], self.Charge + 1, 'zero')
            for n in range(self.nzBlocks.shape[0]):
                B.Blocks[self.BlockPos(self.nzBlocks[n, 0], 1 + self.nzBlocks[n, 1])] = self.Blocks[n]
        elif dir == 'r':
            B = U1SymMat([dim] + self.SizeR + [dim], self.SizeC, self.Charge - 1, 'zero')
            for n in range(self.nzBlocks.shape[0]):
                B.Blocks[self.BlockPos(1 + self.nzBlocks[n, 0], self.nzBlocks[n, 1])] = self.Blocks[n]
        return B

    def __add__(self, other):
        if len(self.nzBlocks) == 0:
            return other
        elif len(other.nzBlocks) == 0:
            return self
        else:
            if self.Charge == other.Charge:
                A = self
                for n in range(other.nzBlocks.shape[0]):
                    if self.Blocks[n] is None:
                        A.Blocks[n] = other.Blocks[n]
                    else:
                        if other.Blocks[n] is not None:
                            A.Blocks[n] = self.Blocks[n] + other.Blocks[n]
                return A
            else:
                raise ValueError('Charges must be equal')

    def __neg__(self):
        if len(self.nzBlocks) == 0:
            return self
        else:
            A = U1SymMat(self.SizeR, self.SizeC, self.Charge, 'null')
            for n in range(len(self.nzBlocks)):
                A.Blocks[n] = -self.Blocks[n]
            return A

    def __sub__(self, other):
        if len(self.nzBlocks) == 0:
            return -other
        elif len(other.nzBlocks) == 0:
            return self
        else:
            if self.Charge == other.Charge:
                A = self
                for n in range(other.nzBlocks.shape[0]):
                    if self.Blocks[n] is None:
                        A.Blocks[n] = -other.Blocks[n]
                    else:
                        if other.Blocks[n] is not None:
                            A.Blocks[n] = self.Blocks[n] - other.Blocks[n]
                return A
            else:
                raise ValueError('Charges must be equal')

    def transpose(self):
        if len(self.nzBlocks) == 0:
            return self
        else:
            A = U1SymMat(self.SizeC, self.SizeR, -self.Charge, 'null')
            for n in range(len(self.nzBlocks)):
                A.Blocks[n] = self.Blocks[n].T
            return A

    def conj(self):
        if len(self.nzBlocks) == 0:
            return self
        else:
            A = U1SymMat(self.SizeR, self.SizeC, self.Charge, 'null')
            for n in range(len(self.nzBlocks)):
                A.Blocks[n] = np.conj(self.Blocks[n])
            return A

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if abs(other) < np.finfo(float).eps:
                return U1SymMat(self.SizeR, self.SizeC, self.Charge, 'empty')
            elif len(self.nzBlocks) == 0:
                return U1SymMat(self.SizeR, self.SizeC, self.Charge, 'empty')
            else:
                A = U1SymMat(self.SizeR, self.SizeC, self.Charge, 'null')
                for n in range(len(self.nzBlocks)):
                    if self.Blocks[n] is not None:
                        A.Blocks[n] = other * self.Blocks[n]
                return A
        else:
            if len(self.nzBlocks) == 0:
                return U1SymMat(self.SizeR, other.SizeC, self.Charge + other.Charge, 'empty')
            elif len(other.nzBlocks) == 0:
                return U1SymMat(other.SizeR, self.SizeC, self.Charge + other.Charge, 'empty')
            else:
                A = U1SymMat(self.SizeR, other.SizeC, self.Charge + other.Charge, 'null')
                Start = max(self.nzBlocks[0, 1], other.nzBlocks[0, 0])
                End = min(self.nzBlocks[-1, 1], other.nzBlocks[-1, 0])

                B1 = np.where(self.nzBlocks[:, 1] == Start)[0]
                C1 = np.where(other.nzBlocks[:, 0] == Start)[0]

                A1 = np.where(A.nzBlocks[:, 0] == self.nzBlocks[B1[0], 0])[0]

                for n in range(End - Start + 1):
                    if self.Blocks[B1[n]] is None or other.Blocks[C1[n]] is None:
                        A.Blocks[A1[n]] = None
                    else:
                        A.Blocks[A1[n]] = np.dot(self.Blocks[B1[n]], other.Blocks[C1[n]])
                return A

    def dinv(self, tol):
        B = U1SymMat(self.SizeR, self.SizeC, self.Charge, 'null')
        for n in range(len(self.nzBlocks)):
            s = self.Blocks[n].shape[0]
            B.Blocks[n] = np.zeros((s, s))
            for k in range(s):
                if self.Blocks[n][k, k] > tol:
                    B.Blocks[n][k, k] = self.Blocks[n][k, k] ** (-1)
        return B

    def pinv(self, tol):
        B = U1SymMat(self.SizeR, self.SizeC, self.Charge, 'null')
        for n in range(len(self.nzBlocks)):
            try:
                u, g, v = np.linalg.svd(self.Blocks[n])
            except np.linalg.LinAlgError:
                print('svd did not converge in pinv, add hash and try again')
                self.Blocks[n] = self.Blocks[n] + 1.e-10 * (
                    np.random.rand(self.Blocks[n].shape[0], self.Blocks[n].shape[1])) + \
                                 1j * np.random.rand(self.Blocks[n].shape[0], self.Blocks[n].shape[1])
                u, g, v = np.linalg.svd(self.Blocks[n])
            for k in range(g.shape[0]):
                if g[k] < tol:
                    g[k] = 0
                else:
                    g[k] = 1 / g[k]
            B.Blocks[n] = np.dot(np.dot(v.T, np.diag(g)), u.T)
        return B

    def expm(self):
        B = U1SymMat(self.SizeR, self.SizeC, self.Charge, 'null')
        for n in range(len(self.nzBlocks)):
            B.Blocks[n] = np.linalg.expm(self.Blocks[n])
        return B

    def sqrt(self):
        B = U1SymMat(self.SizeR, self.SizeC, self.Charge, 'null')
        for n in range(len(self.nzBlocks)):
            B.Blocks[n] = np.sqrt(self.Blocks[n])
        return B

    def svd(self):
        U = U1SymMat(self.SizeR, self.SizeR, 0, 'null')
        D = U1SymMat(self.SizeR, self.SizeR, 0, 'null')
        V = U1SymMat(self.SizeR, self.SizeR, 0, 'null')

        for n in range(len(self.nzBlocks)):
            U.Blocks[n], D.Blocks[n], V.Blocks[n] = np.linalg.svd(self.Blocks[n], full_matrices=False)

        return U, D, V

    def norm(self):
        o = 0
        for n in range(len(self.nzBlocks)):
            o += np.linalg.norm(self.Blocks[n])
        return o

    def eig(self):
        U = U1SymMat(self.SizeR, self.SizeR, 0, 'null')
        g = U1SymMat(self.SizeR, self.SizeR, 0, 'null')

        for n in range(len(self.nzBlocks)):
            U.Blocks[n], g.Blocks[n] = np.linalg.eig(self.Blocks[n])
            ind = np.argsort(np.real(g.Blocks[n]))
            g.Blocks[n] = np.real(g.Blocks[n][ind])
            U.Blocks[n] = U.Blocks[n][:, ind]
        return U, g

    def trace(self):
        if len(self.SizeR) == len(self.SizeC) and self.Charge == 0:
            a = 0
            for n in range(len(self.nzBlocks)):
                a += np.trace(self.Blocks[n])
            return a
        else:
            raise ValueError('Cannot calculate trace')

    def trace2(self, C):
        C.nzBlocks = np.flip(C.nzBlocks, axis=1)

        Start = max(self.nzBlocks[0, 1], C.nzBlocks[0, 0])
        End = min(self.nzBlocks[-1, 1], C.nzBlocks[-1, 0])

        B1 = np.where(self.nzBlocks[:, 1] == Start)[0]
        C1 = np.where(C.nzBlocks[:, 0] == Start)[0]

        a = 0
        for n in range(End - Start + 1):
            a += np.dot(self.Blocks[B1[n]].reshape(1, -1), C.Blocks[C1[n]].reshape(-1, 1))[0, 0]
        return a

    def __abs__(self):
        for n in range(len(self.nzBlocks)):
            self.Blocks[n] = np.abs(self.Blocks[n])
        return self

    def max(self):
        a = self.Blocks[0][0, 0]
        for n in range(len(self.nzBlocks)):
            m = np.max(self.Blocks[n])
            if a < m:
                a = m
        return a
