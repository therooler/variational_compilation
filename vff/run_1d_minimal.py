import torch
from torch import optim

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
import warnings

warnings.simplefilter("ignore", UserWarning)

# Everything here is based on QUIMB and PyTorch
from tn.hamiltonians import get_hamiltonian
from training.utils import training_loop
from tn.mps_circuit import TNModel, qmps_brick, qmps_brick_quasi_1d, create_targets, load_gates
from tn.data_states import random_product_state, \
    random_mps_state, get_make_data_set_fn, random_U1_state
from tn.trotter import compress_trotterization_into_circuit



def is_su4_format(s):
    return re.fullmatch(r'SU4_[-]?\d+_[-]?\d+', s) is not None


def get_circuit_dict(pqc, num_layers):
    # Convert to unitaries
    pqc = pqc.isometrize(method='qr')
    circuit_dict = {}
    for l in range(num_layers):
        circuit_dict[f"{l}"] = {}
        tensors = pqc[(f"L{l}")]
        if not isinstance(tensors, tuple):
            tensors = (tensors,)
        for t in tensors:
            name = list(filter(lambda x: is_su4_format(x), t.tags))[0]
            circuit_dict[f"{l}"][name] = t.data.detach().numpy()
    print("Saved circuit:")
    return circuit_dict


def main(config):
    # 0.) CONFIG PARSING

    # META PARAMS
    TRAIN = config['TRAIN']
    PLOT = config['PLOT']
    PRINT = config.get('PRINT', True)
    SHOW = config['SHOW']
    TEST = config['TEST']
    TEST_UNITARY = config['TEST_UNITARY']
    HST = config.get('HST', False)
    SCRATCH_PATH = config.get('SCRATCH_PATH', None)
    SEED = config['SEED']
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    GET_PATH = config.get('GET_PATH', False)

    test_size = config.get('test_size', 100)
    # Model properties
    if 'L' in config.keys():
        L = config['L']
    else:
        Lx, Ly = config['Lx'], config['Ly']
        L = Lx * Ly
    hamiltonian = config['hamiltonian']
    t = config['t']

    trotter_start = config.get('trotter_start', False)
    trotter_start_order = config.get('trotter_start_order', 1)
    # assert
    L, H, hamiltonian_path, trotter_initialization, get_Utrotter, bc = get_hamiltonian(config)
    if PRINT:
        print(f"Model {hamiltonian} of size {L} at time {t:1.3f}\n")
    granularity_from_t = int(np.round(t / 0.05))
    tebd_granularity = config.get('tebd_granularity', granularity_from_t)  # How fine is the TEBD grid
    tebd_cutoff = config.get('tebd_cutoff', -1)  # What is the SVD cutoff for TEBD
    tebd_max_bond = config.get('max_bond', 20)  # What is the bond dimension cutoff for TEBD
    tebd_opts = {'cutoff': tebd_cutoff, 'max_bond': tebd_max_bond}
    ctg = config.get('ctg', False)
    if PRINT:
        print(f"TEBD steps: {tebd_granularity}")
        print(f"TEBD max bond dimension: {tebd_max_bond}")
        print(f"TEBD cutoff: {tebd_cutoff}\n")

    # Training properties
    circuit_name = config['circuit_name']  # What circuit ansatz
    circuit_translation = config.get('circuit_translation', False)
    training_states = config['training_states']  # Type of training states
    num_steps = config['num_steps']  # Number of max iterations in training
    num_samples = config['num_samples']  # Number of training states

    make_data_set = get_make_data_set_fn(hamiltonian, H(L), tebd_granularity, tebd_opts, PRINT)

    assert num_samples > 0
    training_strategy = config['training_strategy']  # Strategy for optimizing the circuit
    get_training_state = lambda x, s: random_product_state(x, s)
    training_state_path = f"{training_states}"

    depth_min = config.get('depth_min')
    depth_max = config.get('depth_max')
    depth_step = config['depth_step']

    training_strategy_path = f"{training_strategy}_dstep_{depth_step}"
    path_name = f'UNITARY_COMPILATION/{hamiltonian_path}/L_{L}/' \
                f'{training_strategy_path}/t_{t:1.3f}/Nsteps_{num_steps}_' \
                f'{circuit_name}' \
                f'{"_translation" if circuit_translation else ""}' \
                f'{"_trotter_init_" if trotter_start else ""}' \
                f'{trotter_start_order if trotter_start else ""}/' \
                f'SEED_{SEED}/Ns_{num_samples}_{training_state_path}_bd_max_{tebd_max_bond}'
    if PRINT:
        print(f"(hotstart) Min depth = {depth_min}")
        print(f"(hotstart) Max depth = {depth_max}")
        print(f"(hotstart) Depth step= {depth_step}")
        print(f"(hotstart) Trotter start= {trotter_start}")
        print(f"(hotstart) Trotter p = {trotter_start_order}\n")

    learning_rate = config['learning_rate']  # Learning rate schedule
    learning_rate_scheduler = config['learning_rate_schedule']  # Learning rate schedule
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
    elif TRAIN:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + 'config'):
            with open(save_path + 'config', 'w') as file:
                for k, v in config.items():
                    file.write(f'{k} = {v}\n')
        # Create or Load Train data
        if not os.path.exists(save_path + 'psi0.pickle') or not os.path.exists(save_path + 'psit.pickle'):

            print("Creating training data set")
            if not hamiltonian == 'heisenberg_2d':
                psi0_list_train, psit_list_train, tebd_errors = make_data_set(lambda x: get_training_state(L, x), t,
                                                                              num_samples, SEED)
            else:
                psi0_list_train, psit_list_train, tebd_errors = make_data_set(
                    lambda x: get_training_state(Lx * Ly, x), t,
                    num_samples, SEED)
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

        depth_list = list(range(depth_min, depth_max + 1, depth_step))
        for d_i, depth in enumerate(depth_list):
            save_path_depth = save_path + f"depth_{depth}/"
            save_path_depth_previous = save_path + f"depth_{depth - depth_step}/"
            save_path_depth_ckpts = save_path_depth + "ckpts/"
            save_path_depth_ckpts_previous = save_path_depth_previous + "ckpts/"
            if not os.path.exists(save_path_depth + "train_loss.npy"):

                if not os.path.exists(save_path_depth_ckpts):
                    os.makedirs(save_path_depth_ckpts)
                # # If we're in the first layer, start with the identity, otherwise, perturb.
                if (depth == depth_min) and trotter_start:
                    tn = trotter_initialization(-t, 1)
                    new_tn = compress_trotterization_into_circuit(L, tn)
                    start_even = 'Even' in new_tn.tensors[0].tags
                    num_layers = len(list(filter(lambda x: bool(re.match(r'^L\d+$', x)), new_tn.tags)))
                    assert num_layers == depth_min, f"Minmial number of layers is {depth_min}," \
                                                    f" but must be equal to the number of layers " \
                                                    f" in the trotterization: {num_layers}"
                    psi_pqc = qmps_brick(L, in_depth=num_layers, rand=False, val_iden=0.01,
                                         start_even=start_even)
                    load_gates(psi_pqc, new_tn, transpose=True)
                else:
                    psi_pqc = qmps_brick(L, in_depth=depth, rand=False, val_iden=0.01)


                psi, psi_tars = create_targets(L, psi_pqc, psi0_list_train, psit_list_train, device=device)

                model = TNModel(psi, psi_tars, translation=circuit_translation, ctg=ctg)
                if depth > depth_min:
                    # Load previous parameters, strict=false means we don't need the number of parameters to match
                    model.load_state_dict(torch.load(save_path_depth_ckpts_previous + "parameters.ckpt",
                                                     map_location=torch.device(device)), strict=False)
                model.eval()
                print(f"Start loss = {model.forward():1.12f}")
                model.to(device)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        action='ignore',
                        message='.*trace might not generalize.*',
                    )
                    model = torch.jit.trace_module(model, {"forward": []})
                lr = learning_rate
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = learning_rate_scheduler(optimizer)

                model = training_loop(optimizer, model, scheduler, num_steps, show_progress=PRINT)

                # Save the model state.
                torch.save(model.state_dict(), save_path_depth_ckpts + "parameters.ckpt")

                final_loss = model().cpu().detach().numpy()
                np.save(save_path_depth + 'train_loss', final_loss)

                # Save to normal format:
                circuit_dict = get_circuit_dict(psi, num_layers)
                with open(save_path_depth + 'circuit.c','wb') as file:
                    pickle.dump(circuit_dict, file)
                np.save(save_path_depth + 'train_loss', final_loss)
            else:
                previous_loss = np.load(save_path_depth + 'train_loss.npy')
                print(f"Data for depth {depth} exists, loss = {previous_loss}")
            if TEST:
                if not os.path.exists(save_path_depth + f'test_loss_{test_size}.npy'):
                    # Create or Load test data
                    if not os.path.exists(save_path + 'psi0_test.pickle') or not os.path.exists(
                            save_path + 'psit_test.pickle'):
                        print("Creating test dataset")
                        psi0_list_test, psit_list_test, tebd_errors = make_data_set(
                            lambda x: get_training_state(L, x), t,
                            test_size, SEED + 10 ** 6)
                        np.save(save_path + 'tebd_errors_test', np.array(tebd_errors))
                        with open(save_path + 'psi0_test.pickle', 'wb') as file:
                            pickle.dump(psi0_list_test, file)
                        with open(save_path + 'psit_test.pickle', 'wb') as file:
                            pickle.dump(psit_list_test, file)
                    else:
                        with open(save_path + 'psi0_test.pickle', 'rb') as file:
                            psi0_list_test = pickle.load(file)
                        with open(save_path + 'psit_test.pickle', 'rb') as file:
                            psit_list_test = pickle.load(file)
                        tebd_errors = np.load(save_path + 'tebd_errors_test.npy')
                        print("Restored test dataset from file")
                        assert len(psi0_list_test) == test_size
                        assert len(psit_list_test) == test_size
                    print(f"TEBD error: {np.mean(tebd_errors)}")
                    # If we're in the first layer, start with the identity, otherwise, perturb.
                    if hamiltonian != 'heisenberg_2d':
                        psi_pqc = qmps_brick(L, in_depth=depth, rand=False, val_iden=0.01)
                    else:
                        psi_pqc = qmps_brick(Lx, Ly, in_depth=depth, rand=False, val_iden=0.01,
                                              boundary_condition=bc)
                    psi, psi_tars = create_targets(L, psi_pqc, psi0_list_test, psit_list_test, device=device)
                    model = TNModel(psi, psi_tars, translation=circuit_translation, ctg=ctg)
                    # Load previous parameters,
                    # strict=false means we don't need the number of parameters to match
                    model.load_state_dict(torch.load(save_path_depth_ckpts + "parameters.ckpt",
                                                     map_location=torch.device(device)), strict=True)
                    model.eval()
                    model.to(device)
                    test_loss = model().cpu().detach().numpy()
                    np.save(save_path_depth + f'test_loss_{test_size}', test_loss)
                else:
                    test_loss = np.load(save_path_depth + f'test_loss_{test_size}.npy')

                print(f"Test loss for {test_size} samples = {test_loss}")


if __name__ == '__main__':
    config = {
        # MODEL
        'L': 16,
        'hamiltonian': 'ising',
        't': 0.5,
        # TEBD
        'test_size': 100,  # number of states in test ensemble
        'tebd_test_steps': 20,  # number of steps in TEBD
        'max_bond': None,  # maximum bond dimension in tebd
        'tebd_cutoff': 1e-10,  # maximum bond dimension in tebd
        # 'ctg': True,
        # TRAINING
        'circuit_name': 'brickwall',
        'circuit_translation': False,  # translation invariant circuit
        'num_steps': 1000,  # maximum number of optimization steps
        'num_samples': 1,  # number of training samples
        'training_states': 'product',
        # STRATEGY`
        'training_strategy': 'hotstart',
        # HOTSTART
        'depth_max': 10,  # maximum circuit depth
        'depth_min': 5,  # increase depth per step
        'depth_step': 1,  # increase depth per step
        'trotter_start': True,  # increase depth per step
        'trotter_start_order': 2,  # increase depth per step
        # OPTIMIZATION
        'learning_rate': 0.001,
        'learning_rate_schedule': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5),
        # META
        'TRAIN': True,
        'TEST': True,
        'TEST_UNITARY': False,
        'PLOT': False,
        'PLOT': False,
        'SHOW': False,
        'SEED': 0
    }
    main(config)
