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
import matplotlib.pyplot as plt

import tqdm
import os
import quimb as qu
import random
import pickle
import warnings

warnings.simplefilter("ignore", UserWarning)

from utils.plot_utils import plot_losses
# Everything here is based on QUIMB and PyTorch
from tn.peps_circuit import TNModel, qmps_brick_2d, create_targets

import quimb.tensor as qtn

def peps_evo(Lx, Ly, ham, t, trotter_steps, chi, seed=111):
    psi0 = qtn.PEPS.rand(Lx, Ly, bond_dim=2, dtype=float, seed=seed)
    psi0 = psi0 / (psi0.norm())
    psit = psi0.copy()
    su = qtn.SimpleUpdate(
        psit,
        ham,
        D=chi,
        compute_energy_final=False,
        compute_energy_per_site=False,
    )
    #TODO: Technically real-time is not supported, but we do it anyway
    su.evolve(trotter_steps, 1j * t / trotter_steps * 4, progbar=False)
    # psit = su.state.normalize()
    # psit = su.state / np.sqrt((su.state.H & su.state).contract(optimize='auto-hq'))
    psit = su.state / np.sqrt((su.state.H & su.state).contract_boundary(max_bond=chi))
    return psi0, psit


def main(config):
    # 0.) CONFIG PARSING

    # META PARAMS
    TRAIN = config['TRAIN']
    PLOT = config['PLOT']
    PRINT = config.get('PRINT', True)
    SHOW = config['SHOW']
    TEST = config['TEST']
    SCRATCH_PATH = config.get('SCRATCH_PATH', None)
    test_size = config.get('test_size', 100)
    SEED = config['SEED']
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    GET_PATH = config.get('GET_PATH', False)

    # Model properties
    Lx = config['Lx']
    Ly = config['Ly']
    L = Lx * Ly

    hamiltonian = config['hamiltonian']
    t = config['t']

    if hamiltonian == 'ising':
        g = config.get('g', 1.0)
        ham = qu.ham_ising(2, bx=g)
        hamiltonian_path = f"{hamiltonian}_g_{g:1.2f}"

    elif hamiltonian == 'heisenberg':
        ham = qu.ham_heis(2)
        hamiltonian_path = f"{hamiltonian}"
    else:
        raise NotImplementedError
    if PRINT:
        print(f"Model {hamiltonian} of size {Lx}x{Ly} at time {t:1.3f}\n")
    granularity_from_t = int(np.round(t / 0.05))
    tebd_granularity = config.get('tebd_granularity', granularity_from_t)  # How fine is the TEBD grid
    tebd_max_bond = config.get('max_bond', 20)  # What is the bond dimension cutoff for TEBD
    if PRINT:
        print(f"TEBD steps: {granularity_from_t}")
        print(f"TEBD max bond dimension: {tebd_max_bond}\n")

    # Training properties
    circuit_name = config['circuit_name']  # What circuit ansatz
    circuit_translation = config.get('circuit_translation', False)
    training_states = config['training_states']  # Type of training states
    if hamiltonian == 'ising_nnn':
        assert training_states in ['product', 'mps'], "Only random product states are supported for NNN MPS."
    num_steps = config['num_steps']  # Number of max iterations in training
    num_samples = config['num_samples']  # Number of training states
    training_strategy = config['training_strategy']
    training_state_path = 'bd_2'
    assert num_samples > 0

    if training_strategy == 'hotstart':
        depth_min = config.get('depth_min')
        depth_max = config.get('depth_max')
        depth_step = config['depth_step']

        training_strategy_path = f"{training_strategy}_dstep_{depth_step}"
        path_name = f'UNITARY_COMPILATION/{hamiltonian_path}/{Lx}x{Ly}/' \
                    f'{training_strategy_path}/t_{t:1.3f}/Nsteps_{num_steps}_' \
                    f'{circuit_name}' \
                    f'{"_translation" if circuit_translation else ""}/' \
                    f'SEED_{SEED}/Ns_{num_samples}_{training_state_path}_bd_max_{tebd_max_bond}'
        if PRINT:
            print(f"(hotstart) Min depth = {depth_min}")
            print(f"(hotstart) Max depth = {depth_max}")
            print(f"(hotstart) Depth step= {depth_step}")

    else:
        raise NotImplementedError
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
        fig_path = SCRATCH_PATH + f'/data/{path_name}/figures/'
    else:
        save_path = f'./data/{path_name}/'
        fig_path = f'./data/{path_name}/figures/'

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
            psi0_list_train = []
            psit_list_train = []
            for i in range(num_samples):
                psi0, psit = peps_evo(Lx, Ly, qtn.LocalHam2D(Lx, Ly, H2=ham), t, tebd_granularity, tebd_max_bond, seed=SEED + i)
                psi0_list_train.append(psi0)
                psit_list_train.append(psit)
            with open(save_path + 'psi0.pickle', 'wb') as file:
                pickle.dump(psi0_list_train, file)
            with open(save_path + 'psit.pickle', 'wb') as file:
                pickle.dump(psit_list_train, file)
        else:
            with open(save_path + 'psi0.pickle', 'rb') as file:
                psi0_list_train = pickle.load(file)
            with open(save_path + 'psit.pickle', 'rb') as file:
                psit_list_train = pickle.load(file)
            print("Restored train dataset from file")
            assert len(psi0_list_train) == num_samples
            assert len(psit_list_train) == num_samples

        if training_strategy == 'hotstart':
            depth_list = list(range(depth_min, depth_max + 1, depth_step))
            for d_i, depth in enumerate(depth_list):
                save_path_depth = save_path + f"depth_{depth}/"
                save_path_depth_previous = save_path + f"depth_{depth - depth_step}/"
                save_path_depth_ckpts = save_path_depth + "ckpts/"
                save_path_depth_ckpts_previous = save_path_depth_previous + "ckpts/"
                if not os.path.exists(save_path_depth + "train_loss.npy"):
                    if not os.path.exists(save_path_depth_ckpts):
                        os.makedirs(save_path_depth_ckpts)
                    psi_pqc = qmps_brick_2d(Lx, Ly, in_depth=depth, rand=False, val_iden=0.01)
                    psi, psi_tars = create_targets(L, psi_pqc, psi0_list_train, psit_list_train)
                    # model = TNModel(psi, psi_tars, device, contract_opts={'max_bond': tebd_max_bond})
                    model = TNModel(psi, psi_tars, device)

                    if depth > depth_min:
                        # Load previous parameters, strict=false means we don't need the number of parameters to match
                        model.load_state_dict(torch.load(save_path_depth_ckpts_previous + "parameters.ckpt"),
                                              strict=False)
                    model.eval()
                    print(f"Start loss = {model.forward():1.12f}")
                    model.to(device)
                    lr = learning_rate
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    scheduler = learning_rate_scheduler(optimizer)

                    pbar = tqdm.tqdm(range(num_steps), disable=not PRINT)
                    previous_loss = torch.inf
                    for step in pbar:
                        optimizer.zero_grad()
                        loss = model.forward()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        pbar.set_description(f"Loss={loss} - LR={scheduler.get_last_lr()[0]}")
                        if step > 100 and torch.abs(previous_loss - loss) < 1e-10:
                            print("Early stopping loss difference is smaller than 1e-10")
                            break
                        previous_loss = loss.clone()

                    # Save the model state.
                    torch.save(model.state_dict(), save_path_depth_ckpts + "parameters.ckpt")
                    final_loss = model().cpu().detach().numpy()
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
                            psi0_list_test = []
                            psit_list_test = []
                            for i in range(test_size):
                                psi0, psit = peps_evo(Lx, Ly, qtn.LocalHam2D(Lx, Ly, H2=ham), t, tebd_granularity, tebd_max_bond,
                                                      seed=SEED + 10 ** 6 + i)
                                psi0_list_test.append(psi0)
                                psit_list_test.append(psit)

                            with open(save_path + 'psi0_test.pickle', 'wb') as file:
                                pickle.dump(psi0_list_test, file)
                            with open(save_path + 'psit_test.pickle', 'wb') as file:
                                pickle.dump(psit_list_test, file)
                        else:
                            with open(save_path + 'psi0_test.pickle', 'rb') as file:
                                psi0_list_test = pickle.load(file)
                            with open(save_path + 'psit_test.pickle', 'rb') as file:
                                psit_list_test = pickle.load(file)
                            print("Restored test dataset from file")
                            assert len(psi0_list_test) == test_size
                            assert len(psit_list_test) == test_size
                        # If we're in the first layer, start with the identity, otherwise, perturb.
                        psi_pqc = qmps_brick_2d(Lx, Ly, in_depth=depth, rand=False, val_iden=0.01)
                        psi, psi_tars = create_targets(L, psi_pqc, psi0_list_test, psit_list_test)
                        model = TNModel(psi, psi_tars, device)
                        # Load previous parameters,
                        # strict=false means we don't need the number of parameters to match
                        model.load_state_dict(torch.load(save_path_depth_ckpts + "parameters.ckpt"),
                                              strict=True)
                        model.eval()
                        model.to(device)
                        test_loss = model().cpu().detach().numpy()
                        np.save(save_path_depth + f'test_loss_{test_size}', test_loss)
                    else:
                        test_loss = np.load(save_path_depth + f'test_loss_{test_size}.npy')

                    print(f"Test loss for {test_size} samples = {test_loss}")

    if PLOT:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        if training_strategy == 'hotstart':
            train_losses = []
            test_losses = []
            unitary_losses = []
            for depth in depth_list:
                save_path_depth = save_path + f"depth_{depth}/"
                train_losses.append(np.load(save_path_depth + 'train_loss.npy'))
                try:
                    test_losses.append(np.load(save_path_depth + f'test_loss_{test_size}.npy'))
                except FileNotFoundError:
                    test_losses.append(np.nan)
                try:
                    unitary_losses.append(np.load(save_path_depth + 'unitary_loss.npy'))
                except FileNotFoundError:
                    unitary_losses.append(np.nan)
            train_losses = np.array(train_losses).flatten()
            test_losses = np.array(test_losses).flatten()
            unitary_losses = np.array(unitary_losses).flatten()

            plot_losses(depth_list, train_losses, test_losses, unitary_losses, fig_path)
            if SHOW:
                plt.show()
