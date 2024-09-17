import numpy as np
import quimb.tensor as qtn


def apply_tebd(psi, H, ts, tebd_opts, tol=1e-5, dt=None, PRINT=True):
    tebd = qtn.TEBD(psi, H, split_opts=tebd_opts, dt=dt, progbar=PRINT)
    for psit in tebd.at_times(ts, tol=tol):
        pass
    return psit, tebd.err


def ising_hamiltonian_quimb(L: int, jxx: float, jz: float) -> qtn.LocalHam1D:
    builder = qtn.SpinHam1D(S=1 / 2, cyclic=False)
    # specify the interaction term (defaults to all sites)
    builder += 4.0 * jxx, 'x', 'x'
    builder += 2.0 * jz, 'z'
    return builder.build_local_ham(L)


def longitudinal_ising_hamiltonian_quimb(L: int, jxx: float, jz: float, jx: float) -> qtn.LocalHam1D:
    builder = qtn.SpinHam1D(S=1 / 2, cyclic=False)
    # specify the interaction term (defaults to all sites)
    builder += 4.0 * jxx, 'x', 'x'
    builder += 2.0 * jz, 'z'
    builder += 2.0 * jx, 'x'
    return builder.build_local_ham(L)


def heisenberg_hamiltonian_quimb(L: int) -> qtn.LocalHam1D:
    builder = qtn.SpinHam1D(S=1 / 2, cyclic=False)
    # specify the interaction term (defaults to all sites)
    builder += 4.0, 'x', 'x'
    builder += 4.0, 'y', 'y'
    builder += 4.0, 'z', 'z'
    return builder.build_local_ham(L)


def mbl_hamiltonian_quimb(L: int, delta: float, dh: np.ndarray) -> qtn.LocalHam1D:
    builder = qtn.SpinHam1D(S=1 / 2, cyclic=False)
    # specify the interaction term (defaults to all sites)
    builder += 4.0, 'x', 'x'
    builder += 4.0, 'y', 'y'
    builder += delta * 4.0, 'z', 'z'
    for i in range(L):
        builder[i] += 2 * dh[i], 'z'
    return builder.build_local_ham(L)
