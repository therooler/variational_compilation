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
import warnings

warnings.simplefilter("ignore", UserWarning)

# Everything here is based on QUIMB and PyTorch
from tn.data_states import random_product_state, \
    random_mps_state, get_make_data_set_fn, random_U1_state
from tn.tebd import ising_hamiltonian_quimb, \
    longitudinal_ising_hamiltonian_quimb, \
    heisenberg_hamiltonian_quimb, \
    mbl_hamiltonian_quimb


def main(config):
    # 0.) CONFIG PARSING

    # META PARAMS
    PRINT = config.get('PRINT', True)
    SCRATCH_PATH = config.get('SCRATCH_PATH', None)

    SEED = config['SEED']
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    GET_PATH = config.get('GET_PATH', False)

    # Model properties
    L = config['L']
    hamiltonian = config['hamiltonian']
    t = config['t']

    if hamiltonian == 'ising':
        g = config.get('g', 1.0)
        H = ising_hamiltonian_quimb(L, 1.0, g)
        hamiltonian_path = f"{hamiltonian}_g_{g:1.2f}"

    elif hamiltonian == 'longitudinal_ising':
        jx = config.get('jx', 1.0)
        jz = config.get('jz', 1.0)
        H = longitudinal_ising_hamiltonian_quimb(L, 1.0, jz, jx)
        hamiltonian_path = f"{hamiltonian}_jz_{jz:1.2f}_jx_{jx:1.2f}"

    elif hamiltonian == 'heisenberg':
        H = heisenberg_hamiltonian_quimb(L)
        hamiltonian_path = f"{hamiltonian}"

    elif hamiltonian == 'mbl':
        sigma = config.get('sigma', 1.0)
        delta = config.get('delta', 1.0)
        dh = (2.0 * np.random.rand(L) - 1.0) * sigma
        H = mbl_hamiltonian_quimb(L, delta, dh)
        hamiltonian_path = f"{hamiltonian}_sigma_{sigma:1.3f}"

    elif hamiltonian == 'ising_nnn':
        J = config.get('J', -1.0)
        V = config.get('V', 1.0)
        # define the model, notice the extra factor of 4 and 2
        assert not L % 2, "Number of sites must be even for NNN Ising"

        if GET_PATH:
            H = None
        else:
            H = (L, J, V)
        hamiltonian_path = f"{hamiltonian}_J_{J:1.3f}_V_{V:1.3f}"

    else:
        raise NotImplementedError
    if PRINT:
        print(f"Model {hamiltonian} of size {L} at time {t:1.3f}\n")
    granularity_from_t = int(np.round(t / 0.05))
    tebd_granularity = config.get('tebd_granularity', granularity_from_t)  # How fine is the TEBD grid
    tebd_cutoff = config.get('tebd_cutoff', -1)  # What is the SVD cutoff for TEBD
    tebd_max_bond = config.get('max_bond', 20)  # What is the bond dimension cutoff for TEBD
    tebd_opts = {'cutoff': tebd_cutoff, 'max_bond': tebd_max_bond}
    if PRINT:
        print(f"TEBD steps: {granularity_from_t}")
        print(f"TEBD max bond dimension: {tebd_max_bond}\n")

    # Training properties
    circuit_name = config['circuit_name']  # What circuit ansatz
    circuit_translation = config.get('circuit_translation', False)
    training_states = config['training_states']  # Type of training states
    num_steps = config['num_steps']  # Number of max iterations in training
    num_samples = config['num_samples']  # Number of training states

    make_data_set = get_make_data_set_fn(hamiltonian, H, tebd_granularity, tebd_opts, PRINT)

    assert num_samples > 0
    training_strategy = config['training_strategy']  # Strategy for optimizing the circuit
    if training_states == 'product':
        get_training_state = lambda s: random_product_state(L, s)
        training_state_path = f"{training_states}"
    elif training_states == 'mps':
        train_state_bond_dim = config.get('train_state_bond_dim', 2)
        get_training_state = lambda s: random_mps_state(L, train_state_bond_dim, s)
        training_state_path = f"mps_bd_{train_state_bond_dim}"
    elif training_states == 'u1':
        num_particles = config.get('num_particles')
        train_state_bond_dim = config.get('train_state_bond_dim', 3)
        assert train_state_bond_dim > 2, 'If train state bond dimension is <= 2 all states are equal.'
        get_training_state = lambda s: random_U1_state(L, train_state_bond_dim, num_particles, s)
        training_state_path = f"u1_bd_{train_state_bond_dim}_np_{num_particles}"
    else:
        raise NotImplementedError

    training_strategy_path = f"tebd_bd"
    path_name = f'TEBD_BD/{hamiltonian_path}/L_{L}/' \
                f'{training_strategy_path}/t_{t:1.3f}/SEED_{SEED}/{training_state_path}'

    if PRINT:
        print(f"Circuit: {circuit_name}")
        print(f"Translation: {circuit_translation}")
        print(f"Type of training state: {training_states}")
        print(f"Number of steps: {num_steps}")
        print(f"Number of samples: {num_samples}")
        print(f"Training strategy: {training_strategy}\n")

    ### 2.) Paths ###
    if SCRATCH_PATH is not None:
        save_path = SCRATCH_PATH + f'/data/{path_name}/'
    else:
        save_path = f'./data/{path_name}/'

    if GET_PATH:
        return save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path + 'config'):
        with open(save_path + 'config', 'w') as file:
            for k, v in config.items():
                file.write(f'{k} = {v}\n')

        # Create or Load Train data
        if not os.path.exists(save_path + 'psi0.pickle') or not os.path.exists(save_path + 'psit.pickle'):

            print("Creating training data set")
            psi0_list_train, psit_list_train, tebd_errors = make_data_set(get_training_state, t, num_samples, SEED)
            with open(save_path + 'psi0.pickle', 'wb') as file:
                pickle.dump(psi0_list_train, file)
            with open(save_path + 'psit.pickle', 'wb') as file:
                pickle.dump(psit_list_train, file)
            np.save(save_path + 'tebd_errors_train', np.array(tebd_errors))
        else:
            with open(save_path + 'psi0.pickle', 'rb') as file:
                psi0_list_train = pickle.load(file)
            with open(save_path + 'psit.pickle', 'rb') as file:
                psit_list_train = pickle.load(file)
            tebd_errors = np.load(save_path + 'tebd_errors_train.npy')
            print("Restored train dataset from file")
            assert len(psi0_list_train) == num_samples
            assert len(psit_list_train) == num_samples
        print(f"TEBD error: {np.mean(tebd_errors)}")
        save_path = save_path + f"max_bds.npy"
        bond_dims = []
        for psi in psit_list_train:
            bond_dims.append(max(psi.bond_sizes()))
        if not os.path.exists(save_path):
            np.save(save_path, np.array(bond_dims))
        else:
            bond_dims = np.load(save_path)
        print("Max bond dimensions: ", bond_dims)
