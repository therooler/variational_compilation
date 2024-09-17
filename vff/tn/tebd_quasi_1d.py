import numpy as np
import scipy.linalg as spla

import networkx as nx
import matplotlib.pyplot as plt

Px = np.array([[0, 1], [1, 0]])
Py = np.array([[0, -1j], [1j, 0]])
Pz = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])


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


def quasi_1d_tebd_heisenberg(psi, Lx, Ly, t, trotter_steps, cutoff=1e-8, reindex=True, boundary_condition=(False, False)):
    split_opts = {}
    split_opts['cutoff'] = cutoff
    dic_2d_1d = snake_index(Lx, Ly)
    H_i = np.kron(Px, Px) + np.kron(Py, Py) + np.kron(Pz, Pz)
    G = spla.expm(-1j * H_i * t / trotter_steps)
    psi_0 = psi.copy()
    if reindex:
        psi_0 = psi_0.reindex({'k' + str(v): 'k' + str(k) for k, v in dic_2d_1d.items()})
    psi_t = psi.copy()
    n_apply = 0
    for _ in range(trotter_steps):
        for r in range(2):
            if r % 2:
                for ly in range(Ly):
                    for lx in range(0, Lx - 1, 2):
                        # print((lx, ly), (lx + 1, ly))
                        psi_t.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx + 1, ly)]), contract='swap+split',
                                    tags={f'SU4_{lx},{ly}_{lx + 1},{ly}', f'G{n_apply}', f'L{r}', 'LR'}, **split_opts)
                        n_apply += 1
                    for lx in range(1, Lx - 1, 2):
                        # print((lx, ly), (lx + 1, ly))
                        psi_t.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx + 1, ly)]), contract='swap+split',
                                    tags={f'SU4_{lx},{ly}_{lx + 1},{ly}', f'G{n_apply}', f'L{r}', 'LR'}, **split_opts)
                        n_apply += 1
                    if boundary_condition[0]:
                        psi_t.gate_(G, (dic_2d_1d[(0, ly)], dic_2d_1d[(Lx-1, ly)]), contract='swap+split',
                                    tags={f'SU4_{0},{ly}_{Lx-1},{ly}', f'G{n_apply}', f'L{r}', 'UD'}, **split_opts)
                        n_apply += 1
            else:
                for lx in range(Lx):
                    for ly in range(0, Ly - 1, 2):  # odd
                        # print((lx, ly), (lx, ly+1))
                        psi_t.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx, ly + 1)]), contract='swap+split',
                                    tags={f'SU4_{lx},{ly}_{lx},{ly + 1}', f'G{n_apply}', f'L{r}', 'UD'}, **split_opts)
                        n_apply += 1
                    for ly in range(1, Ly - 1, 2):  # even
                        # print((lx, ly), (lx, ly+1))
                        psi_t.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx, ly + 1)]), contract='swap+split',
                                    tags={f'SU4_{lx},{ly}_{lx},{ly + 1}', f'G{n_apply}', f'L{r}', 'UD'}, **split_opts)
                        n_apply += 1
                    if boundary_condition[1]:
                        # print((lx, 0), (lx, Ly-1))
                        psi_t.gate_(G, (dic_2d_1d[(lx, 0)], dic_2d_1d[(lx, Ly-1)]), contract='swap+split',
                                    tags={f'SU4_{lx},{0}_{lx},{Ly - 1}', f'G{n_apply}', f'L{r}', 'UD'}, **split_opts)
                        n_apply += 1
    psi_t = psi_t.reindex({'k' + str(v): 'k' + str(k) for k, v in dic_2d_1d.items()})
    return psi_0, psi_t


def quasi_1d_tebd_heisenberg_p2(psi, Lx, Ly, t, trotter_steps, cutoff=1e-8, reindex=True):
    split_opts = {}
    split_opts['cutoff'] = cutoff
    dic_2d_1d = snake_index(Lx, Ly)
    psi_0 = psi.copy()
    if reindex:
        psi_0 = psi_0.reindex({'k' + str(v): 'k' + str(k) for k, v in dic_2d_1d.items()})
    psi_t = psi.copy()
    n_apply = 0
    # a_list = trotter_schemes_per_order[2].a_list
    c_1 = 1. / 2
    a_1 = 1. / 6
    b_1 = 1. / 6 * (3 - np.sqrt(3))
    b_2 = 1. / 2 - b_1
    a_2 = 1. - 2 * a_1
    # A+B+C = XX+YY+ZZ
    a_list = [a_1, b_1, c_1, b_2, a_2, b_2, c_1, b_1, a_1]
    term_list = ['A', 'B', 'C', 'B', 'A', 'B', 'C', 'B', 'A']
    # a_list = [1.0, 1.0, 1.0]
    # G = qu.rand_uni(4, dtype=complex)
    for _ in range(trotter_steps):
        for r in range(len(a_list)):

            if term_list[r] == 'A':
                H_i = np.kron(Px, Px)
            elif term_list[r] == 'B':
                H_i = np.kron(Py, Py)
            elif term_list[r] == 'C':
                H_i = np.kron(Pz, Pz)
            G = spla.expm(-1j * H_i * t / trotter_steps * a_list[r])
            for ly in range(Ly):
                for lx in range(0, Lx - 1, 2):
                    psi_t.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx + 1, ly)]), contract='swap+split',
                                tags={f'SU4_{lx},{ly}_{lx + 1},{ly}', f'G{n_apply}', f'L{r}', 'LR'}, **split_opts)
                    n_apply += 1
                for lx in range(1, Lx - 1, 2):
                    psi_t.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx + 1, ly)]), contract='swap+split',
                                tags={f'SU4_{lx},{ly}_{lx + 1},{ly}', f'G{n_apply}', f'L{r}', 'LR'}, **split_opts)
                    n_apply += 1
            for lx in range(Lx):
                for ly in range(0, Ly - 1, 2):  # odd
                    psi_t.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx, ly + 1)]), contract='swap+split',
                                tags={f'SU4_{lx},{ly}_{lx},{ly + 1}', f'G{n_apply}', f'L{r}', 'UD'}, **split_opts)
                    n_apply += 1
                for ly in range(1, Ly - 1, 2):  # even
                    psi_t.gate_(G, (dic_2d_1d[(lx, ly)], dic_2d_1d[(lx, ly + 1)]), contract='swap+split',
                                tags={f'SU4_{lx},{ly}_{lx},{ly + 1}', f'G{n_apply}', f'L{r}', 'UD'}, **split_opts)
                    n_apply += 1
    psi_t = psi_t.reindex({'k' + str(v): 'k' + str(k) for k, v in dic_2d_1d.items()})
    return psi_0, psi_t


def plot_snaked_2d(Lx, Ly):
    L = Lx * Ly
    dict2d = snake_index(Lx, Ly)
    revdict2d = dict(zip(dict2d.values(), dict2d.keys()))
    graph = nx.Graph()
    pos = []
    for i in range(L):
        graph.add_node(i)
        pos.append(revdict2d[i])
    edges_even = []
    for ly in range(Ly):
        for lx in range(0, Lx - 1, 2):
            edges_even.append((dict2d[(lx, ly)], dict2d[(lx + 1, ly)]))

        for lx in range(1, Lx - 1, 2):
            edges_even.append((dict2d[(lx, ly)], dict2d[(lx + 1, ly)]))
    edges_odd = []

    for lx in range(Lx):
        for ly in range(0, Ly - 1, 2):  # odd
            edges_odd.append((dict2d[(lx, ly)], dict2d[(lx, ly + 1)]))
        for ly in range(1, Ly - 1, 2):  # even
            edges_odd.append((dict2d[(lx, ly)], dict2d[(lx, ly + 1)]))
    nx.draw(graph, pos=pos, with_labels=True, node_color='black', font_color='white')
    odde = nx.Graph()
    odde.add_edges_from(edges_odd)
    evene = nx.Graph()
    evene.add_edges_from(edges_even)
    nx.draw_networkx_edges(odde, pos=dict(zip(range(L), pos)), edge_color='red')
    nx.draw_networkx_edges(evene, pos=dict(zip(range(L), pos)), edge_color='blue')
    plt.show()
