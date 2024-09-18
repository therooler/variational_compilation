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

import pickle
from .tn.data_states import random_product_state, random_mps_state, random_U1_state, \
    plus_state, get_make_data_set_fn
from .tn.mps_circuit import TNModel, qmps_brick, qmps_brick_quasi_1d, create_targets, apply_circuit_to_state, \
    apply_2d_circuit_to_state

from .run_1d import main as main_run_1d
import quimb.tensor as qtn

from .tn.tebd import ising_hamiltonian_quimb, \
    longitudinal_ising_hamiltonian_quimb, \
    heisenberg_hamiltonian_quimb, \
    mbl_hamiltonian_quimb
from .tn.tebd_quasi_1d import snake_index
import random


# For next-nearest neighbors we use Tenpy

def main(config):
    # 0.) CONFIG PARSING

    # META PARAMS
    GET_PATH = config.get('GET_PATH')
    SEED = config['SEED']
    PRINT = config['PRINT']
    training_strategy = config['training_strategy']
    hamiltonian = config['hamiltonian']
    t = config['t']
    max_factor = config['max_factor']
    tebd_cutoff = config.get('tebd_cutoff', -1)  # What is the SVD cutoff for TEBD
    tebd_cutoff_circuit = config.get('tebd_cutoff_circuit', 1e-7)  # What is the SVD cutoff for TEBD
    tebd_max_bond = config.get('max_bond', 20)  # What is the bond dimension cutoff for TEBD
    method = config.get('method', 'double_circuit')  # What is the bond dimension cutoff for TEBD
    tebd_opts = {'cutoff': tebd_cutoff, 'max_bond': tebd_max_bond}
    if 'L' in config.keys():
        L = config['L']
    else:
        Lx, Ly = config['Lx'], config['Ly']
        L = Lx * Ly
    trotter_start = config.get('trotter_start', False)
    trotter_start_order = config.get('trotter_start_order', None)
    circuit_translation = config.get('circuit_translation', False)
    initial_state = config.get('initial_state', 'product')

    # Different choices of initial states
    if isinstance(initial_state, str):
        if initial_state == 'product':
            initial_state_fn = lambda s: random_product_state(L, s)
            training_state_path = f"{initial_state}"
        elif initial_state == 'plus':
            initial_state_fn = lambda s: plus_state(L, s)
            training_state_path = 'mps_plus'
        elif initial_state == 'mps':
            train_state_bond_dim = config.get('initial_state_bond_dim', 2)
            initial_state_fn = lambda s: random_mps_state(L, train_state_bond_dim, s)
            train_state_bond_dim = config.get('train_state_bond_dim', 2)
            training_state_path = f"mps_bd_{train_state_bond_dim}"
        elif initial_state == 'u1':
            num_particles = config.get('num_particles')
            train_state_bond_dim = config.get('train_state_bond_dim', 2)
            initial_state_fn = lambda s: random_U1_state(L, train_state_bond_dim, num_particles, s)
            training_state_path = f"mps_bd_{train_state_bond_dim}"

        else:
            raise NotImplementedError
    elif isinstance(initial_state, qtn.MatrixProductState):
        initial_state_fn = lambda s: initial_state
        training_state_path = "custom_state"

    if PRINT:
        print(f"{hamiltonian} - L={L}")
        if method == 'double_circuit':
            print(f"Max time {2 ** max_factor * t}")
        elif method == 'stacked':
            print(f"Max time {max_factor * t}")
        elif method == 'stacked_old':
            print(f"Max time {max_factor * t}")
        print(f"Time step {t}")

    config_1d = config.copy()
    config_1d['GET_PATH'] = True
    config_1d['PRINT'] = False
    path = main_run_1d(config_1d)
    uni_path = path.split('UNITARY_COMPILATION')[1]
    uni_path = f'./data/EVOLUTION/{uni_path}{method}/{training_state_path}/'
    save_path = f'{uni_path}'
    if GET_PATH:
        return uni_path
    start_depth = config['start_depth']
    if training_strategy == 'hotstart':
        ckpt_path = path + f'depth_{start_depth}/ckpts/'
        loss_path = path + f'depth_{start_depth}/'
    elif training_strategy == 'double_time':
        max_factor = config.get('max_factor')
        initial_depth = config.get('initial_depth')
        factor = 2 ** (max_factor - 1)
        start_depth = initial_depth * factor
        t = t * factor
        ckpt_path = f'{path}t_{t:1.3f}/depth_{start_depth}/ckpts/'
        loss_path = f'{path}t_{t:1.3f}/depth_{start_depth}/'
    else:
        raise NotImplementedError
    assert os.path.exists(ckpt_path + "parameters.ckpt"), f'No variational circuit parameters found at {ckpt_path}\n' \
                                                          f'Run optimization first!'
    test_size = config['test_size']
    if PRINT:
        print(f"Test loss at t {t:1.3f} = {np.load(loss_path + f'test_loss_{test_size}.npy')}")
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if hamiltonian == 'ising':
        g = config.get('g', 1.0)
        H = ising_hamiltonian_quimb(L, 1.0, g)

    elif hamiltonian == 'longitudinal_ising':
        jx = config.get('jz', 1.0)
        jz = config.get('jx', 1.0)
        H = longitudinal_ising_hamiltonian_quimb(L, 1.0, jz, jx)

    elif hamiltonian == 'heisenberg':
        H = heisenberg_hamiltonian_quimb(L)

    elif hamiltonian == 'heisenberg_2d':
        bc = config.get('boundary_condition', (False, False))
        H = (Lx, Ly, bc)

    elif hamiltonian == 'mbl':
        sigma = config.get('sigma', 1.0)
        delta = config.get('delta', 1.0)
        dh = (2.0 * np.random.rand(L) - 1.0) * sigma
        H = mbl_hamiltonian_quimb(L, delta, dh)

    elif hamiltonian == 'ising_nnn':
        J = config.get('J', -1.0)
        V = config.get('V', 1.0)
        # define the model, notice the extra factor of 4 and 2
        assert not L % 2, "Number of sites must be even for NNN Ising"
        H = L, J, V
    else:
        raise NotImplementedError

    if method == 'double_circuit':
        infidelities_tebd = []
        bds_tebd = []
        times = [t * 2 ** f for f in range(max_factor)]
        for i, t_i in enumerate(times):
            save_path_t_i = save_path + f't_{t_i:1.3f}/depth_{start_depth}/'

            if os.path.exists(save_path_t_i + 'infidelity.npy'):
                with open(save_path_t_i + 'psit_var.pickle', 'rb') as file:
                    infidelity_with_tebd = np.load(save_path_t_i + 'infidelity.npy')
                with open(save_path_t_i + 'psit_var.pickle', 'rb') as file:
                    bd_with_tebd = np.load(save_path_t_i + 'bd.npy')
            else:
                print(f"Estimating for t={t_i:1.3f}")
                if not os.path.exists(save_path_t_i):
                    os.makedirs(save_path_t_i)
                depth = start_depth * 2 ** i
                print(f"DEPTH = {depth}")
                granularity_from_t = int(np.round(t_i / 0.05))
                tebd_granularity = config.get('tebd_granularity', granularity_from_t)  # How fine is the TEBD grid
                make_data_set = get_make_data_set_fn(hamiltonian, H, tebd_granularity, tebd_opts, PRINT)

                if not os.path.exists(save_path_t_i + 'psi0.pickle') or not os.path.exists(
                        save_path_t_i + 'psit.pickle'):
                    print("Creating training data set")
                    psi0_list_train, psit_list_train, tebd_errors = make_data_set(initial_state_fn, t_i, 1,
                                                                                  SEED)
                    with open(save_path_t_i + 'psi0.pickle', 'wb') as file:
                        pickle.dump(psi0_list_train, file)
                    with open(save_path_t_i + 'psit.pickle', 'wb') as file:
                        pickle.dump(psit_list_train, file)
                    np.save(save_path_t_i + 'tebd_errors_train', np.array(tebd_errors))
                else:
                    with open(save_path_t_i + 'psi0.pickle', 'rb') as file:
                        psi0_list_train = pickle.load(file)
                    with open(save_path_t_i + 'psit.pickle', 'rb') as file:
                        psit_list_train = pickle.load(file)
                    tebd_errors = np.load(save_path_t_i + 'tebd_errors_train.npy')
                    print("Restored train dataset from file")

                # # If we're in the first layer, start with the identity, otherwise, perturb.
                if hamiltonian != 'heisenberg_2d':
                    psi_pqc = qmps_brick(L, in_depth=depth, rand=False, val_iden=0.01)
                else:
                    psi_pqc = qmps_brick_quasi_1d(Lx, Ly, in_depth=depth, rand=False, val_iden=0.01, boundary_condition=bc)
                psi, psi_tars = create_targets(L, psi_pqc, psi0_list_train, psit_list_train, device=device)
                model = TNModel(psi, psi_tars, translation=circuit_translation)
                torch_params = torch.load(ckpt_path + "parameters.ckpt", map_location=torch.device(device))
                new_torch_params = dict()
                for c_i in range(2 ** i):
                    for k, v in torch_params.items():
                        index = int(k.split(".")[-1])
                        new_torch_params[f"torch_params.{index + c_i * len(torch_params)}"] = v.clone()
                model.load_state_dict(new_torch_params, strict=True)

                psi0 = psi0_list_train[0]
                if hamiltonian != 'heisenberg_2d':
                    psit_var = apply_circuit_to_state(L, model, psi0, {'cutoff': tebd_cutoff_circuit, 'max_bond': -1},
                                                      translation=circuit_translation)
                else:
                    psit_var = apply_2d_circuit_to_state(Lx, Ly, model, psi0,
                                                         {'cutoff': tebd_cutoff_circuit, 'max_bond': -1},
                                                         translation=circuit_translation)

                psit_tebd = psit_list_train[0]
                if not os.path.exists(save_path_t_i + 'infidelity.npy'):
                    infidelity_with_tebd = 1 - abs((psit_tebd.H & psit_var) ^ all) ** 2
                    np.save(save_path_t_i + 'infidelity.npy', infidelity_with_tebd)
                else:
                    infidelity_with_tebd = np.load(save_path_t_i + 'infidelity.npy')

                if not os.path.exists(save_path_t_i + 'bd.npy'):
                    bd_with_tebd = np.array(psit_tebd.max_bond())
                    np.save(save_path_t_i + 'bd.npy', bd_with_tebd)
                else:
                    bd_with_tebd = np.load(save_path_t_i + 'bd.npy')
                with open(save_path_t_i + 'psit_var.pickle', 'wb') as file:
                    pickle.dump(psit_var, file)
            if PRINT:
                print(f"Infidelity with tebd = {infidelity_with_tebd}")
            infidelities_tebd.append(infidelity_with_tebd)
            bds_tebd.append(bd_with_tebd)
        return times, bds_tebd, infidelities_tebd
    elif method == 'stacked':
        infidelities_tebd = []
        bds_tebd = []
        times = [t * f for f in range(1, max_factor + 1)]
        for i, t_i in enumerate(times):
            save_path_t_i = save_path + f't_{t_i:1.3f}/depth_{start_depth}/'

            if os.path.exists(save_path_t_i + 'infidelity.npy'):
                with open(save_path_t_i + 'psit_var.pickle', 'rb') as file:
                    infidelity_with_tebd = np.load(save_path_t_i + 'infidelity.npy')
                with open(save_path_t_i + 'psit_var.pickle', 'rb') as file:
                    bd_with_tebd = np.load(save_path_t_i + 'bd.npy')
            else:
                print(f"Estimating for t={t_i:1.3f}")
                if not os.path.exists(save_path_t_i):
                    os.makedirs(save_path_t_i)
                depth = start_depth
                print(f"DEPTH = {depth}")
                granularity_from_t = int(np.round(t / 0.05))
                tebd_granularity = config.get('tebd_granularity', granularity_from_t)  # How fine is the TEBD grid
                make_data_set = get_make_data_set_fn(hamiltonian, H, tebd_granularity, tebd_opts, PRINT)
                if i == 0:
                    if not os.path.exists(save_path_t_i + 'psi0.pickle') or not os.path.exists(
                            save_path_t_i + 'psit.pickle'):
                        print("Creating training data set")
                        psi0_var_list_train, psit_list_train, tebd_errors = make_data_set(initial_state_fn, t, 1,
                                                                                          SEED)
                        with open(save_path_t_i + 'psi0.pickle', 'wb') as file:
                            pickle.dump(psi0_var_list_train, file)
                        with open(save_path_t_i + 'psit.pickle', 'wb') as file:
                            pickle.dump(psit_list_train, file)
                        np.save(save_path_t_i + 'tebd_errors_train', np.array(tebd_errors))
                    else:
                        with open(save_path_t_i + 'psi0.pickle', 'rb') as file:
                            psi0_var_list_train = pickle.load(file)
                        with open(save_path_t_i + 'psit.pickle', 'rb') as file:
                            psit_list_train = pickle.load(file)
                        tebd_errors = np.load(save_path_t_i + 'tebd_errors_train.npy')
                        print("Restored train dataset from file")
                else:
                    # load previous state
                    save_path_t_i_previous = save_path + f't_{t_i - t:1.3f}/depth_{start_depth}/'
                    with open(save_path_t_i_previous + 'psit.pickle', 'rb') as file:
                        psi0_list_train = pickle.load(file)
                    with open(save_path_t_i_previous + 'psit_var.pickle', 'rb') as file:
                        psi0_var_list_train = pickle.load(file)
                    if hamiltonian == 'heisenberg_2d':
                        dic_2d_1d = snake_index(Lx, Ly)
                        dic_1d_2d = dict(zip(dic_2d_1d.values(), dic_2d_1d.keys()))
                        for _i in range(L):
                            tn = psi0_list_train[0][_i]
                            tn.reindex({f"k{dic_1d_2d[_i]}": f"k{_i}"}, inplace=True)

                    _, psit_list_train, tebd_errors = make_data_set(lambda s: psi0_list_train[0], t, 1,
                                                                    SEED)
                # # If we're in the first layer, start with the identity, otherwise, perturb.
                if hamiltonian != 'heisenberg_2d':
                    psi_pqc = qmps_brick(L, in_depth=depth, rand=False, val_iden=0.01)
                else:
                    psi_pqc = qmps_brick_quasi_1d(Lx, Ly, in_depth=depth, rand=False, val_iden=0.01, boundary_condition=bc)
                psi, psi_tars = create_targets(L, psi_pqc, psi0_var_list_train, psit_list_train, device=device)
                model = TNModel(psi, psi_tars, translation=circuit_translation)
                torch_params = torch.load(ckpt_path + "parameters.ckpt", map_location=torch.device(device))
                model.load_state_dict(torch_params, strict=True)

                psi0 = psi0_var_list_train[0]
                psi0.compress(**tebd_opts)
                if hamiltonian != 'heisenberg_2d':
                    psit_var = apply_circuit_to_state(L, model, psi0, {'cutoff': tebd_cutoff_circuit, 'max_bond': -1},
                                                      translation=circuit_translation)
                else:
                    psit_var = apply_2d_circuit_to_state(Lx, Ly, model, psi0,
                                                         {'cutoff': tebd_cutoff_circuit, 'max_bond': -1},
                                                         translation=circuit_translation)

                psit_tebd = psit_list_train[0]
                if not os.path.exists(save_path_t_i + 'infidelity.npy'):
                    infidelity_with_tebd = 1 - abs((psit_tebd.H & psit_var) ^ all) ** 2
                    print('Done')
                    np.save(save_path_t_i + 'infidelity.npy', infidelity_with_tebd)
                else:
                    infidelity_with_tebd = np.load(save_path_t_i + 'infidelity.npy')

                if not os.path.exists(save_path_t_i + 'bd.npy'):
                    bd_with_tebd = np.array(psit_tebd.max_bond())
                    np.save(save_path_t_i + 'bd.npy', bd_with_tebd)
                else:
                    bd_with_tebd = np.load(save_path_t_i + 'bd.npy')
                with open(save_path_t_i + 'psit_var.pickle', 'wb') as file:
                    pickle.dump([psit_var, ], file)
                with open(save_path_t_i + 'psit.pickle', 'wb') as file:
                    pickle.dump([psit_tebd, ], file)
                print("BD", bd_with_tebd)
                print("BD var", psit_var.max_bond())
            if PRINT:
                print(f"Infidelity with tebd = {infidelity_with_tebd}")
            infidelities_tebd.append(infidelity_with_tebd)
            bds_tebd.append(bd_with_tebd)
        return times, bds_tebd, infidelities_tebd
    else:
        raise NotImplementedError
