import torch

CUDA_ENABLED = torch.cuda.is_available()
print(f"Is GPU available? {CUDA_ENABLED}")
if CUDA_ENABLED:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(f"Device {device}")

import numpy as np

import os
import random
import pickle
import re

from tn.data_states import random_product_state, random_mps_state, get_make_data_set_fn
from tn.tebd import ising_hamiltonian_quimb, \
    longitudinal_ising_hamiltonian_quimb, \
    heisenberg_hamiltonian_quimb, \
    mbl_hamiltonian_quimb
from tn.mps_circuit import TNModel, qmps_brick, create_targets, load_gates, apply_circuit_to_state
from tn.trotter import trotter_evolution_optimized_nn_ising_tn, \
    trotter_evolution_optimized_nn_heisenberg_tn, trotter_evolution_optimized_nn_mbl_tn, \
    trotter_evolution_optimized_ising_nnn_tn, compress_trotterization_into_circuit
from tn.tebd_quasi_1d import quasi_1d_tebd_heisenberg, quasi_1d_tebd_heisenberg_p2, snake_index
from utils.ed import build_TFIZ_matrix, build_heisenberg_matrix, build_mbl_matrix, build_SDIsing_matrix
import quimb.tensor as qtn

import scipy


def main(config):
    # 0.) CONFIG PARSING

    # META PARAMS
    DATA = config['DATA']
    EXACT = config['EXACT']
    PRINT = config['PRINT']
    SEED = config['SEED']
    SCRATCH_PATH = config.get('SCRATCH_PATH', None)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    GET_PATH = config.get('GET_PATH', False)

    # Model properties
    if 'L' in config.keys():
        L = config['L']
    else:
        Lx, Ly = config['Lx'], config['Ly']
        L = Lx * Ly
    hamiltonian = config['hamiltonian']
    t = config['t']
    test_size = config.get('test_size', 100)
    tebd_test_steps = config.get('tebd_test_steps', 20)
    trotter_order = config.get('trotter_order', 1)
    if PRINT:
        print(f"Model {hamiltonian} of size {L} at time {t:1.3f}\n")
        print(f"Trotter order {trotter_order}")
    training_states = config['training_states']  # Type of training states
    if training_states == 'product':
        get_training_state = lambda s: random_product_state(L, s)

        training_state_path = f"{training_states}"
    elif training_states == 'mps':
        train_state_bond_dim = config.get('train_state_bond_dim', 2)
        get_training_state = lambda s: random_mps_state(L, train_state_bond_dim, s)
        training_state_path = f"mps_bd_{train_state_bond_dim}"
    else:
        raise NotImplementedError
    # Exact TEBD states
    granularity_from_t = int(np.round(t / 0.05))
    tebd_granularity = config.get('tebd_granularity', granularity_from_t)  # How fine is the TEBD grid
    tebd_cutoff = config.get('tebd_cutoff', 1e-9)  # What is the SVD cutoff for TEBD
    tebd_max_bond = config.get('max_bond', None)  # What is the bond dimension cutoff for TEBD
    tebd_opts_exact = {'cutoff': tebd_cutoff, 'max_bond': tebd_max_bond}
    final_contraction_bd = config.get('final_contraction_bd',-1) # What is the SVD cutoff for TEBD
    # Trotterized TEBD states
    m = config.get('m', 2)
    dt = t / m

    if hamiltonian == 'ising':
        g = config.get('g', 1.0)
        H = ising_hamiltonian_quimb(L, 1.0, g)
        hamiltonian_path = f"{hamiltonian}_g_{g:1.2f}"
        get_H_matrix = lambda: build_TFIZ_matrix(L, Jxx=1.0, Jz=g, Jx=0.0)
        get_U_tebd = lambda: trotter_evolution_optimized_nn_ising_tn(L, 1.0, g, 0.0, t / tebd_test_steps,
                                                                     tebd_test_steps, p=2)
        get_U_trotter = lambda: trotter_evolution_optimized_nn_ising_tn(L, 1.0, g, 0.0, dt, m, p=trotter_order)
    elif hamiltonian == 'longitudinal_ising':
        jx = config.get('jx', 1.0)
        jz = config.get('jz', 1.0)
        H = longitudinal_ising_hamiltonian_quimb(L, 1.0, jz, jx)
        hamiltonian_path = f"{hamiltonian}_jz_{jz:1.2f}_jx_{jx:1.2f}"
        get_H_matrix = lambda: build_TFIZ_matrix(L, Jxx=1.0, Jz=jz, Jx=jx)
        get_U_tebd = lambda: trotter_evolution_optimized_nn_ising_tn(L, 1.0, jz, jx, t / tebd_test_steps,
                                                                     tebd_test_steps, p=2)
        get_U_trotter = lambda: trotter_evolution_optimized_nn_ising_tn(L, 1.0, jz, jx, dt, m, p=trotter_order)
    elif hamiltonian == 'heisenberg':
        H = heisenberg_hamiltonian_quimb(L)
        hamiltonian_path = f"{hamiltonian}"
        get_H_matrix = lambda: build_heisenberg_matrix(L)
        get_U_tebd = lambda: trotter_evolution_optimized_nn_heisenberg_tn(L, t / tebd_test_steps,
                                                                          tebd_test_steps, p=2)
        get_U_trotter = lambda: trotter_evolution_optimized_nn_heisenberg_tn(L, dt, m, p=trotter_order)
    elif hamiltonian == 'heisenberg_2d':
        H = (Lx, Ly)
        hamiltonian_path = f"{hamiltonian}/{Lx}x{Ly}"
        get_H_matrix = lambda s: NotImplementedError
        get_U_trotter = lambda: None
    elif hamiltonian == 'mbl':
        sigma = config.get('sigma', 1.0)
        delta = config.get('delta', 1.0)
        dh = (2.0 * np.random.rand(L) - 1.0) * sigma
        H = mbl_hamiltonian_quimb(L, delta, dh)
        hamiltonian_path = f"{hamiltonian}_sigma_{sigma:1.3f}"
        get_H_matrix = lambda: build_mbl_matrix(L, delta, dh)
        get_U_tebd = lambda: trotter_evolution_optimized_nn_mbl_tn(L, delta, dh, t / tebd_test_steps,
                                                                   tebd_test_steps, p=2)
        get_U_trotter = lambda: trotter_evolution_optimized_nn_mbl_tn(L, delta, dh, dt, m, p=trotter_order)
    elif hamiltonian == 'ising_nnn':
        J = config.get('J', -1.0)
        V = config.get('V', 1.0)
        hamiltonian_path = f"{hamiltonian}_J_{J:1.3f}_V_{V:1.3f}"
        # define the model, notice the extra factor of 4 and 2
        assert not L % 2, "Number of sites must be even for NNN Ising"

        H = L, J,V
        get_H_matrix = lambda:  build_SDIsing_matrix(L, J, V)
        get_U_tebd = lambda: NotImplementedError
        get_U_trotter = lambda: trotter_evolution_optimized_ising_nnn_tn(L, J, V, dt, m, p=trotter_order)
        if not GET_PATH:
            get_U_trotter()

    else:
        raise NotImplementedError

    ### 2.) Paths ###
    path_name = f'TROTTERIZATION/{hamiltonian_path}/L_{L}/t_{t:1.3f}/' \
                f'SEED_{SEED}/Ns_{test_size}_{training_state_path}_bd_max_{tebd_max_bond}'

    if SCRATCH_PATH is not None:
        save_path = SCRATCH_PATH + f'/data/{path_name}/'
        trotter_path = SCRATCH_PATH + f'/data/{path_name}/order_{trotter_order}/m_{m}/'
    else:
        save_path = f'./data/{path_name}/'
        trotter_path = f'./data/{path_name}/order_{trotter_order}/m_{m}/'

    if GET_PATH:
        return save_path
    elif DATA:
        make_data_set = get_make_data_set_fn(hamiltonian, H, tebd_granularity, tebd_opts_exact, PRINT)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + 'config'):
            with open(save_path + 'config', 'w') as file:
                for k, v in config.items():
                    file.write(f'{k} = {v}\n')
        # Create test set
        if not os.path.exists(save_path + 'psi0.pickle') or not os.path.exists(save_path + 'psit.pickle'):
            psi0_list_test, psit_list_test, tebd_errors = make_data_set(get_training_state, t, test_size, SEED)
            with open(save_path + 'psi0.pickle', 'wb') as file:
                pickle.dump(psi0_list_test, file)
            with open(save_path + 'psit.pickle', 'wb') as file:
                pickle.dump(psit_list_test, file)
            np.save(save_path + 'tebd_errors_test', np.array(tebd_errors))
        else:
            with open(save_path + 'psi0.pickle', 'rb') as file:
                psi0_list_test = pickle.load(file)
            with open(save_path + 'psit.pickle', 'rb') as file:
                psit_list_test = pickle.load(file)
            print("Restored test dataset from file")
            assert len(psi0_list_test) == test_size
            assert len(psit_list_test) == test_size
        U_trotter_tn = None
        if EXACT:
            if not os.path.exists(trotter_path + 'tebd_loss.npy'):
                U_trotter_tn = get_U_trotter()
                if not os.path.exists(trotter_path):
                    os.makedirs(trotter_path)
                U_tebd_tn = get_U_tebd()
                U_exact_tensor = scipy.linalg.expm(1j * get_H_matrix().todense() * t)
                U_exact_tensor = qtn.Tensor(U_exact_tensor.reshape([2] * (2 * L)),
                                            tuple([f'b{i}' for i in range(L)] + [f'k{i}' for i in range(L)]))
                exact_loss = 1 - (abs((U_exact_tensor.H & U_tebd_tn) ^ all) / 2 ** L) ** 2
                tebd_loss = 1 - (abs((U_trotter_tn.H & U_tebd_tn) ^ all) / 2 ** L) ** 2
                np.save(trotter_path + 'exact_loss', exact_loss)
                np.save(trotter_path + 'tebd_loss', tebd_loss)
            else:
                exact_loss = np.load(trotter_path + 'exact_loss.npy')
                tebd_loss = np.load(trotter_path + 'tebd_loss.npy')
            print(f"Exact loss: {exact_loss}")
            print(f"TEBD loss: {tebd_loss}")
        else:
            tebd_loss = np.nan
        if not os.path.exists(trotter_path + 'test_loss.npy'):
            if not os.path.exists(trotter_path):
                os.makedirs(trotter_path)
            if hamiltonian == 'heisenberg_2d':
                dict2d = snake_index(Lx, Ly)
                _losses = []
                for _psi0, _psit in zip(psi0_list_test, psit_list_test):
                    _psi0 = _psi0.reindex({'k' + str(k): 'k' + str(v) for k, v in dict2d.items()})
                    if trotter_order == 1:
                        _, _psivar = quasi_1d_tebd_heisenberg(_psi0, Lx, Ly, t, m, cutoff=tebd_cutoff, reindex=True)
                    elif trotter_order == 2:
                        _, _psivar = quasi_1d_tebd_heisenberg_p2(_psi0, Lx, Ly, t, m, cutoff=tebd_cutoff, reindex=True)
                    else:
                        raise NotImplementedError
                    _losses.append(1 - abs((_psit.H & _psivar) ^ all) ** 2)
                test_loss = np.mean(_losses)
                print('apply_circuit', np.mean(_losses))
            else:
                if U_trotter_tn is None:
                    U_trotter_tn = get_U_trotter()
                new_tn = compress_trotterization_into_circuit(L, U_trotter_tn)

                num_layers = len(list(filter(lambda x: bool(re.match(r'^L\d+$', x)), new_tn.tags)))
                start_even = 'Even' in new_tn.tensors[0].tags
                psi_pqc = qmps_brick(L, in_depth=num_layers, rand=False, val_iden=0.01,
                                     start_even=start_even)
                load_gates(psi_pqc, new_tn.H, transpose=True)

                psi, psi_tars = create_targets(L, psi_pqc, psi0_list_test, psit_list_test, device=device)
                model = TNModel(psi, psi_tars, contract_opts={'max_bond': final_contraction_bd})
                # model.eval()
                # test_loss = model.forward().detach().numpy()
                # print('model: ', test_loss)
                _losses = []
                psit_var = apply_circuit_to_state(L, model, psi0_list_test, tebd_opts_exact)
                for _psivar, _psit in zip(psit_var, psit_list_test):
                    _losses.append(1 - abs((_psit.H & _psivar) ^ all) ** 2)
                test_loss = np.mean(_losses)
                print('apply_circuit', np.mean(_losses))

            np.save(trotter_path + 'test_loss', test_loss)
        else:
            test_loss = np.load(trotter_path + 'test_loss.npy')
        print(f"Test loss: {test_loss}")
        return tebd_loss, test_loss


if __name__ == '__main__':
    Lx, Ly = 2, 8
    t = 0.2
    m = 2
    p = 2
    config = {
        # MODEL
        'Lx': Lx,
        'Ly': Ly,
        # 'hamiltonian': 'mbl',
        'hamiltonian': 'heisenberg_2d',
        't': t,
        'jx': 1.0,
        'sigma': 5,
        'jz': 1.0,
        'J': -1,
        'V': 0.1,
        'm': m,
        'trotter_order': p,
        'test_size': 5,
        # TEBD
        'tebd_granularity': 100,
        'tebd_test_steps': 20,  # number of steps in TEBD
        'max_bond': None,  # maximum bond dimension in tebd
        'tebd_cutoff': 1e-9,  # set compression cutoff, -1 means max_bd is leading, good value 1e-10
        'final_contraction_cutoff': 1e-9,  # set compression cutoff, -1 means max_bd is leading, good value 1e-10
        'final_contraction_bd': 14,  # set compression cutoff, -1 means max_bd is leading, good value 1e-10
        # DATA
        'training_states': 'product',
        # TROTTER
        'DATA': True,
        'EXACT': False,
        'PRINT': True,
        'SEED': 1
    }
    main(config)
