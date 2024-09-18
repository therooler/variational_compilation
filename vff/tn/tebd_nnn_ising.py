#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:44:44 2024
(requires version: physics-tenpy-1.0.0)
"""
import numpy as np
import scipy.linalg as spla
from .trotter import TwoTermsOrder1, TwoTermsOrder2, TwoTermsOrder4, TwoTermsOrder6, I, Px, Py, Pz


def tebd_ising_nnn(psi, L, t, J, V, trotter_steps, cutoff=1e-8, p=2):
    split_opts = {}
    split_opts['cutoff'] = cutoff
    psi_0 = psi.copy()
    psi_t = psi.copy()
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
    depth = 0
    n_apply = 0
    dt = t / trotter_steps
    for n in range(trotter_steps):
        for r in range(len(a_list)):
            if r % 2 == pattern:
                for i in range(1, L - 1, 1):
                    if i == 1:  # add edge
                        H_i = np.kron(Px, Px)
                        G_J = spla.expm(-1j * J * H_i * dt * a_list[r])
                        G_V = spla.expm(-1j * V * H_i * dt * a_list[r])
                        # J, XX: index_list[i - 1], index_list[i]
                        # J, XX: index_list[i], index_list[i + 1]
                        # V, XIX: index_list[i-1], index_list[i + 1]
                        psi_t.gate_(G_J, (i - 1, i), contract='swap+split',
                                    tags={f'SU4_{i - 1},{i}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
                        psi_t.gate_(G_J, (i, i + 1), contract='swap+split',
                                    tags={f'SU4_{i},{i + 1}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
                        psi_t.gate_(G_V, (i - 1, i + 1), contract='swap+split',
                                    tags={f'SU4_{i - 1},{i + 1}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
                    else:
                        H_i = np.kron(Px, Px)
                        G_J = spla.expm(-1j * J * H_i * dt * a_list[r])
                        G_V = spla.expm(-1j * V * H_i * dt * a_list[r])
                        # J, XX: index_list[i], index_list[i + 1]
                        # V, XIX: index_list[i-1], index_list[i + 1]
                        psi_t.gate_(G_J, (i, i + 1), contract='swap+split',
                                    tags={f'SU4_{i},{i + 1}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
                        psi_t.gate_(G_V, (i - 1, i + 1), contract='swap+split',
                                    tags={f'SU4_{i - 1},{i + 1}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1

            else:
                for i in range(1, L - 1, 1):
                    if i == 1:  # add edge
                        G_J = spla.expm(-1j * J * np.kron(Pz, I) * dt * a_list[r])
                        G_V = spla.expm(-1j * V * np.kron(Pz, Pz) * dt * a_list[r])
                        # J, Z: index_list[i]
                        # J, Z: index_list[i-1]
                        # V, ZZ: index_list[i-1], index_list[i]
                        # V, ZZ: index_list[i], index_list[i + 1]
                        psi_t.gate_(G_J, (i, i + 1), contract='swap+split',
                                    tags={f'SU4_{i},{i + 1}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
                        psi_t.gate_(G_J, (i - 1, i), contract='swap+split',
                                    tags={f'SU4_{i - 1},{i}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
                        psi_t.gate_(G_V, (i - 1, i), contract='swap+split',
                                    tags={f'SU4_{i - 1},{i}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
                        psi_t.gate_(G_V, (i, i + 1), contract='swap+split',
                                    tags={f'SU4_{i},{i + 1}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1

                    elif i == (L - 2):
                        G_J = spla.expm(-1j * J * np.kron(I, Pz) * dt * a_list[r])
                        G_V = spla.expm(-1j * V * np.kron(Pz, Pz) * dt * a_list[r])
                        # J, Z: index_list[i]
                        # J, Z: index_list[i+1]
                        # V, ZZ: index_list[i-1], index_list[i]
                        # V, ZZ: index_list[i], index_list[i + 1]
                        psi_t.gate_(G_J, (i - 1, i), contract='swap+split',
                                    tags={f'SU4_{i - 1},{i}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
                        psi_t.gate_(G_J, (i, i + 1), contract='swap+split',
                                    tags={f'SU4_{i},{i + 1}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
                        psi_t.gate_(G_V, (i, i + 1), contract='swap+split',
                                    tags={f'SU4_{i},{i + 1}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
                    else:
                        G_J = spla.expm(-1j * J * np.kron(Pz, I) * dt * a_list[r])
                        G_V = spla.expm(-1j * V * np.kron(Pz, Pz) * dt * a_list[r])
                        # J, Z: index_list[i]
                        # V, ZZ: index_list[i], index_list[i + 1]
                        psi_t.gate_(G_J, (i, i + 1), contract='swap+split',
                                    tags={f'SU4_{i},{i + 1}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
                        psi_t.gate_(G_V, (i, i + 1), contract='swap+split',
                                    tags={f'SU4_{i},{i + 1}', f'G{n_apply}', f'L{r}'}, **split_opts)
                        n_apply += 1
            depth += 1
    return psi_0, psi_t
