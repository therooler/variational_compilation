import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools as it
import quimb.tensor as qtn

from vff.run_evolution import main as main_evolution
from vff.run_1d import main
from vff.tn.tebd_quasi_1d import snake_index
from vff.tn.data_states import random_U1_state

import torch
import os
import pickle
import quimb


def figure_3a(L):
    config = {
        # MODEL
        'L': L,
        'hamiltonian': 'ising',
        't': 1.,
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
    num_samples_list = [1, 2, 4, 8, 16]
    if GET_DATA:
        for ns in num_samples_list:
            config['num_samples'] = ns
            main(config)
    if PLOT:

        depth_list = list(range(5, 11))
        data = {}
        missing_data = 0
        data_total = 0
        for depth, ns in it.product(depth_list, num_samples_list):
            config['num_samples'] = ns
            config['GET_PATH'] = True
            path = main(config)
            data_total += 1
            try:
                data[f"{ns}_{depth}_train"] = np.load(path + f'depth_{depth}/train_loss.npy')
            except FileNotFoundError:
                data[f"{ns}_{depth}_train"] = np.nan
            try:
                data[f"{ns}_{depth}_test"] = np.load(
                    path + f'depth_{depth}/test_loss_{config["test_size"]}.npy')
            except FileNotFoundError:
                data[f"{ns}_{depth}_test"] = np.nan
                missing_data += 1
        print(f"Data found: {data_total - missing_data}/{data_total}")

        with mpl.rc_context({'font.size': 18, 'font.family': 'serif', "text.usetex": True}):
            MS = 10
            fig, ax = plt.subplots(1, 1)
            ax = [ax, ]
            fig.set_size_inches(6, 5)
            cmap = plt.get_cmap('coolwarm')
            colors = cmap(np.linspace(0., 1., len(num_samples_list)))
            for ns_i, ns in enumerate(num_samples_list):
                losses_train = []
                losses_test = []
                plottable_depths = []
                for depth in depth_list:
                    losses_train.append(data[f"{ns}_{depth}_train"])
                    losses_test.append(data[f"{ns}_{depth}_test"])
                    plottable_depths.append(depth)
                ax[0].plot(plottable_depths, losses_train, color=colors[ns_i], label=rf"$N_s$={ns}", marker='.',
                           markersize=MS)
                ax[0].plot(plottable_depths, losses_test, color=colors[ns_i], linestyle='dashed', marker='.',
                           markersize=MS)
            for axes in ax:
                axes.set_yscale('log')
                axes.set_ylim([1e-7, 1.])
                axes.grid()
                axes.set_xlabel(r'$\tau$')
            ax[0].set_ylabel(r'$C_{\mathcal{D}}$')
            dashed_line = mpl.lines.Line2D([0], [0], color='black', linestyle='--', linewidth=2)
            solid_line = mpl.lines.Line2D([0], [0], color='black', linestyle='-', linewidth=2)
            labels = ['Train', 'Test']
            leg1 = plt.legend(loc='lower left', prop={"size": 15})
            leg2 = plt.legend([solid_line, dashed_line], labels, loc='lower right', prop={"size": 20})
            ax[0].add_artist(leg1)
            ax[0].add_artist(leg2)
            plt.tight_layout()
            fig.savefig(f'./figures/ztfim_{L}_trotter_2.pdf')
            plt.show()


def figure_3b(L):
    config = {
        # MODEL
        'L': L,
        'hamiltonian': 'heisenberg',
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
    num_samples_list = [1, 2, 4, 8, 16]
    if GET_DATA:
        for ns, translation in it.product(num_samples_list, [True, False]):
            config['num_samples'] = ns
            config['circuit_translation'] = translation
            main(config)
    if PLOT:
        depth_list = list(range(5, 11))
        data = {}
        missing_data = 0
        data_total = 0
        for depth, ns, translation in it.product(depth_list, num_samples_list, [True, False]):
            config['circuit_translation'] = translation
            config['num_samples'] = ns
            config['GET_PATH'] = True
            path = main(config)
            data_total += 1
            try:
                data[f"{ns}_{depth}_{translation}_train"] = np.load(path + f'depth_{depth}/train_loss.npy')
            except FileNotFoundError:
                data[f"{ns}_{depth}_{translation}_train"] = np.nan
            try:
                data[f"{ns}_{depth}_{translation}_test"] = np.load(
                    path + f'depth_{depth}/test_loss_{config["test_size"]}.npy')
            except FileNotFoundError:
                data[f"{ns}_{depth}_{translation}_test"] = np.nan
                missing_data += 1
        print(f"Data found: {data_total - missing_data}/{data_total}")

        with mpl.rc_context({'font.size': 18, 'font.family': 'serif', "text.usetex": True}):
            MS = 10
            fig1, ax1 = plt.subplots(1, 2)
            fig1.set_size_inches(5, 5)
            cmap = plt.get_cmap('coolwarm')
            colors = cmap(np.linspace(0., 1., len(num_samples_list)))
            for tr_i, tran in enumerate([True, False]):
                for ns_i, ns in enumerate(num_samples_list):
                    losses_train = []
                    losses_test = []
                    plottable_depths = []
                    for depth in depth_list:
                        losses_train.append(data[f"{ns}_{depth}_{tran}_train"])
                        losses_test.append(data[f"{ns}_{depth}_{tran}_test"])
                        plottable_depths.append(depth)
                    ax1[tr_i].plot(plottable_depths, losses_test, color=colors[ns_i], label=rf"$N_s$={ns}", marker='.',
                                   markersize=MS)
            for axes in ax1:
                axes.set_yscale('log')
                axes.set_ylim([1e-4, 1.])
                axes.grid()
                axes.set_xlabel(r'$\tau$')
                axes.set_xticks(plottable_depths)
            ax1[0].legend(prop={"size": 15})
            ax1[0].set_ylabel(r'$C_{\mathcal{D}_{\mathrm{Test}}}$')
            ax1[0].set_title('TI')
            ax1[1].set_title('No TI')
            plt.tight_layout()
            fig1.subplots_adjust(wspace=0.1)
            ax1[1].set_yticklabels(())
            ax1[1].tick_params(axis="x", which='both', top=False, right=False)
            ax1[1].tick_params(axis="y", which='both', top=False, right=False, left=True)
            fig1.savefig(f'./figures/heisenberg_translation_{L}_trotter_2.pdf')
            plt.show()


def figure_3c():
    config = {
        # MODEL
        'L': 3,
        'hamiltonian': 'ising_nnn',
        'V': 0.2,
        'J': -1.0,
        't': 0.5,
        # TEBD
        'tebd_granularity': 20,
        'max_bond': None,  # maximum bond dimension in tebd
        'tebd_cutoff': 1e-9,  # set compression cutoff, -1 means max_bd is leading, good value 1e-10
        # TRAINING
        'circuit_name': 'brickwall',
        'circuit_translation': True,  # translation invariant circuit
        'training_states': 'product',
        'num_steps': 2000,  # maximum number of optimization steps
        'num_samples': 16,  # number of training samples
        'test_size': 100,
        # Choose training strategy, `hotstart`, `tebd_bd` and `double_time`
        'training_strategy': 'double_space',
        'max_factor': 4,
        'depth': 4,
        'learning_rate': 0.001,
        'learning_rate_schedule': None,
        # META
        'TRAIN': True,
        'TEST': True,
        'TEST_UNITARY': False,
        'PLOT': False,
        'PRINT': False,
        'SHOW': False,
        'SEED': 0
    }
    if GET_DATA:
        for L in [3, 4]:
            config['L'] = L
            main(config)

    if PLOT:
        depth = 4
        data = {}
        missing_data = 0
        data_total = 0
        for L in [3, 4]:
            config['L'] = L
            config['GET_PATH'] = True
            main(config)
            path = main(config)
            for i_L in range(config['max_factor'] + 1):
                data_total += 1
                try:
                    data[f"{i_L}_{L}_train"] = np.load(path + f'L_{L * 2 ** i_L}/depth_{depth}/train_loss.npy')
                except:
                    data[f"{i_L}_{L}_train"] = np.nan
                try:
                    data[f"{i_L}_{L}_test"] = np.load(
                        path + f'L_{L * 2 ** i_L}/depth_{depth}/test_loss_{config["test_size"]}.npy')
                except:
                    data[f"{i_L}_{L}_test"] = np.nan
                    missing_data += 1
        print(f"Data found: {data_total - missing_data}/{data_total}")
        with mpl.rc_context({'font.size': 18, 'font.family': 'serif', "text.usetex": True}):
            MS = 10
            fig, ax = plt.subplots(1, 1)
            ax = [ax]
            fig.set_size_inches(5, 5)
            cmap = plt.get_cmap('coolwarm')
            colors = cmap(np.linspace(0., 1., 2))
            for _L in [3, 4, ]:
                L_list = []
                for i in range(config['max_factor'] + 1):
                    L_list.append(_L * 2 ** i)

                losses_train = []
                losses_test = []
                for i_L, L in enumerate(L_list):
                    losses_train.append(data[f"{i_L}_{_L}_train"])
                    losses_test.append(data[f"{i_L}_{_L}_test"])
                ax[0].plot(L_list, 1 - (1 - np.array(losses_test)) ** (1. / np.array(L_list)),
                           color=colors[-1] if _L == 3 else colors[0],
                           label=r"$n_{\mathrm{init}}$" + rf"=${{{_L}}}$", marker='.' if _L == 4 else '*',
                           markersize=MS)
            for axes in ax:
                axes.set_yscale('log')
                axes.set_ylim([1e-3, 1e-1])
                axes.grid()
                axes.set_xlabel(r'$n$')
            ax[0].set_ylabel(r'$\tilde{C}_{\mathcal{D}_{\mathrm{Test}}}$')
            plt.tight_layout()
            fig.savefig(f'./figures/ising_nnn_per_site.pdf')
            plt.show()


def figure_4a(L):
    config_heis = {
        # MODEL
        'L': L,
        'hamiltonian': 'heisenberg',
        't': 0.1,
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
        'num_samples': 16,  # number of training samples
        'training_states': 'product',
        # STRATEGY`
        'training_strategy': 'hotstart',
        # HOTSTART
        'depth_max': 4,  # maximum circuit depth
        'depth_min': 2,  # increase depth per step
        'depth_step': 1,  # increase depth per step
        'trotter_start': True,  # increase depth per step
        'trotter_start_order': 1,  # increase depth per step
        # OPTIMIZATION
        'learning_rate': 0.001,
        'learning_rate_schedule': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5),
        # META
        'TRAIN': True,
        'TEST': True,
        'TEST_UNITARY': False,
        'PLOT': False,
        'PRINT': False,
        'SHOW': False,
        'SEED': 0
    }
    config_heis['max_factor'] = 200
    config_heis['start_depth'] = 4
    config_heis['method'] = 'stacked'
    config_heis['tebd_cutoff_circuit'] = 1e-7
    inds = [7, 30]
    binary = ['0' if i not in inds else '1' for i in range(L)]
    psi0 = qtn.MPS_computational_state(binary)
    config_heis['initial_state'] = psi0
    if GET_DATA:
        main(config_heis)
        main_evolution(config_heis)
    if PLOT:
        config_heis['GET_PATH'] = True
        t = config_heis['t']
        save_path = main_evolution(config_heis)

        magnetization_per_t_tebd = []
        magnetization_per_t_var = []
        times = [t * (i + 1) for i in range(config_heis['max_factor'])]

        figpath = save_path + 'figures/'
        if not os.path.exists(figpath):
            os.makedirs(figpath)
        if not os.path.exists(save_path + 'data_magnetization_tebd.npy'):
            for i in range(config_heis['max_factor']):
                t_i = t * (i + 1)
                save_path_t_i = save_path + f't_{t_i:1.3f}/depth_{config_heis["start_depth"]}/'
                with open(save_path_t_i + 'psit.pickle', 'rb') as file:
                    psit = pickle.load(file)[0]
                with open(save_path_t_i + 'psit_var.pickle', 'rb') as file:
                    psit_var = pickle.load(file)[0]
                mz_j_tebd = []
                mz_j_var = []
                info = {"cur_orthog": None}
                mz_j_tebd += [psit.magnetization(0, **info)]
                mz_j_var += [psit_var.magnetization(0, **info)]
                for j in range(1, L):
                    mz_j_tebd += [psit.magnetization(j, **info)]
                    mz_j_var += [psit_var.magnetization(j, **info)]
                magnetization_per_t_tebd.append(mz_j_tebd)
                magnetization_per_t_var.append(mz_j_var)
            np.save(save_path + 'data_magnetization_tebd.npy', np.real(magnetization_per_t_tebd))
            np.save(save_path + 'data_magnetization_var.npy', np.real(magnetization_per_t_var))
        magnetization_per_t_tebd = np.load(save_path + 'data_magnetization_tebd.npy')
        magnetization_per_t_var = np.load(save_path + 'data_magnetization_var.npy')
        with mpl.rc_context({'font.size': 18, 'font.family': 'serif', "text.usetex": True}):
            # plot the magnetization
            fig, axs = plt.subplots(1, 2)
            fig.set_size_inches(6, 5)
            plt.set_cmap('Spectral')
            cf1 = axs[0].pcolormesh(np.arange(0, L), times, magnetization_per_t_tebd, vmin=-0.5, vmax=0.5)
            axs[0].set_title('TEBD')
            axs[0].set_xlabel('Site')
            axs[0].set_ylabel('Time T')

            cf2 = axs[1].pcolormesh(np.arange(0, L), times, magnetization_per_t_var, vmin=-0.5, vmax=0.5)
            cbar_ax = fig.add_axes([0.8, 0.17, 0.05, 0.72])
            fig.colorbar(cf2, cax=cbar_ax)
            cbar_ax.set_title(r'$\langle\sigma^z\rangle$')
            # plt.colorbar(cf2)
            axs[1].set_title('Variational')
            axs[1].set_xlabel('Site')
            cf2.axes.set_yticks(())
            axs[1].tick_params(axis="x", which='both', top=False, right=False)
            axs[1].tick_params(axis="y", which='both', top=False, right=False, left=False)
            fig.subplots_adjust(wspace=0.05)
            plt.tight_layout(rect=[0, 0, .8, 1])
            fig.savefig('./figures/heis_lightcone.pdf')
            plt.show()


def figure_4b(L):
    config_mbl = {
        # MODEL
        'L': L,
        'hamiltonian': 'mbl',
        'sigma': 1.0,
        't': 0.1,
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
        'num_samples': 16,  # number of training samples
        'training_states': 'product',
        # STRATEGY`
        'training_strategy': 'hotstart',
        # HOTSTART
        'depth_max': 4,  # maximum circuit depth
        'depth_min': 2,  # increase depth per step
        'depth_step': 1,  # increase depth per step
        'trotter_start': True,  # increase depth per step
        'trotter_start_order': 1,  # increase depth per step
        # OPTIMIZATION
        'learning_rate': 0.001,
        'learning_rate_schedule': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5),
        # META
        'TRAIN': True,
        'TEST': True,
        'TEST_UNITARY': False,
        'PLOT': False,
        'PRINT': False,
        'SHOW': False,
        'SEED': 0
    }
    inds = [7, 30]
    binary = ['0' if i not in inds else '1' for i in range(L)]
    psi0 = qtn.MPS_computational_state(binary)
    config_mbl['max_factor'] = 200
    config_mbl['start_depth'] = 4
    config_mbl['method'] = 'stacked'
    config_mbl['tebd_cutoff_circuit'] = 1e-7
    config_mbl['initial_state'] = psi0
    if GET_DATA:
        main(config_mbl)
        main_evolution(config_mbl)
    if PLOT:
        config_mbl['GET_PATH'] = True
        t = config_mbl['t']
        save_path = main_evolution(config_mbl)

        magnetization_per_t_tebd = []
        magnetization_per_t_var = []
        times = [t * (i + 1) for i in range(config_mbl['max_factor'])]

        figpath = save_path + 'figures/'
        if not os.path.exists(figpath):
            os.makedirs(figpath)
        if not os.path.exists(save_path + 'data_magnetization_tebd.npy'):
            for i in range(config_mbl['max_factor']):
                t_i = t * (i + 1)
                save_path_t_i = save_path + f't_{t_i:1.3f}/depth_{config_mbl["start_depth"]}/'
                with open(save_path_t_i + 'psit.pickle', 'rb') as file:
                    psit = pickle.load(file)[0]
                with open(save_path_t_i + 'psit_var.pickle', 'rb') as file:
                    psit_var = pickle.load(file)[0]
                mz_j_tebd = []
                mz_j_var = []
                info = {"cur_orthog": None}
                mz_j_tebd += [psit.magnetization(0, **info)]
                mz_j_var += [psit_var.magnetization(0, **info)]
                for j in range(1, L):
                    mz_j_tebd += [psit.magnetization(j, **info)]
                    mz_j_var += [psit_var.magnetization(j, **info)]
                magnetization_per_t_tebd.append(mz_j_tebd)
                magnetization_per_t_var.append(mz_j_var)
            np.save(save_path + 'data_magnetization_tebd.npy', np.real(magnetization_per_t_tebd))
            np.save(save_path + 'data_magnetization_var.npy', np.real(magnetization_per_t_var))
        magnetization_per_t_tebd = np.load(save_path + 'data_magnetization_tebd.npy')
        magnetization_per_t_var = np.load(save_path + 'data_magnetization_var.npy')
        with mpl.rc_context({'font.size': 18, 'font.family': 'serif', "text.usetex": True}):
            # plot the magnetization
            fig, axs = plt.subplots(1, 2)
            fig.set_size_inches(6, 5)
            plt.set_cmap('Spectral')
            cf1 = axs[0].pcolormesh(np.arange(0, L), times, magnetization_per_t_tebd, vmin=-0.5, vmax=0.5)
            axs[0].set_title('TEBD')
            axs[0].set_xlabel('Site')
            axs[0].set_ylabel('Time T')

            cf2 = axs[1].pcolormesh(np.arange(0, L), times, magnetization_per_t_var, vmin=-0.5, vmax=0.5)
            cbar_ax = fig.add_axes([0.8, 0.17, 0.05, 0.72])
            fig.colorbar(cf2, cax=cbar_ax)
            cbar_ax.set_title(r'$\langle\sigma^z\rangle$')
            # plt.colorbar(cf2)
            axs[1].set_title('Variational')
            axs[1].set_xlabel('Site')
            cf2.axes.set_yticks(())
            axs[1].tick_params(axis="x", which='both', top=False, right=False)
            axs[1].tick_params(axis="y", which='both', top=False, right=False, left=False)
            fig.subplots_adjust(wspace=0.05)
            plt.tight_layout(rect=[0, 0, .8, 1])
            fig.savefig('./figures/mbl_lightcone.pdf')
            plt.show()


def figure_4c(L):
    config = {
        # MODEL
        'L': L,
        'hamiltonian': 'heisenberg',
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
        'num_samples': 16,  # number of training samples
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
        'HST': True,
        'PLOT': False,
        'PRINT': False,
        'SHOW': False,
        'SEED': 0
    }
    depth_list = list(range(5, 11))
    if GET_DATA:
        main(config)
    if PLOT:
        data = {}
        missing_data = 0
        data_total = 0
        for depth in depth_list:
            config['GET_PATH'] = True
            path = main(config)
            data_total += 1
            try:
                data[f"{depth}_hst"] = np.load(
                    path + f'depth_{depth}/hst_loss.npy')
            except FileNotFoundError:
                data[f"{depth}_hst"] = np.nan
            try:
                data[f"{depth}_test"] = np.load(
                    path + f'depth_{depth}/test_loss_{config["test_size"]}.npy')
            except FileNotFoundError:
                data[f"{depth}_test"] = np.nan
                missing_data += 1
        print(f"Data found: {data_total - missing_data}/{data_total}")

        with mpl.rc_context({'font.size': 18, 'font.family': 'serif', "text.usetex": True}):
            MS = 10
            fig, ax = plt.subplots(1, 1)
            ax = [ax, ]
            fig.set_size_inches(6, 5)
            cmap = plt.get_cmap('coolwarm')
            colors = cmap(np.linspace(0., 1., 2))
            losses_hst = []
            losses_test = []
            plottable_depths = []
            for depth in depth_list:
                losses_hst.append(data[f"{depth}_hst"])
                losses_test.append(data[f"{depth}_test"])
                plottable_depths.append(depth)
            label = None
            color = colors[0]
            ax[0].plot(plottable_depths, np.array(losses_test), color=color, label=label, marker='.',
                       linestyle='dotted', markersize=MS)
            ax[0].plot(plottable_depths, np.array(losses_hst), color=color, linestyle='solid', marker='.',
                       markersize=MS)
            ax[0].plot(plottable_depths, np.array(losses_test) * 2, color=color, linestyle='dashed', marker='.',
                       markersize=MS)
            ax[0].fill_between(plottable_depths, np.array(losses_test), np.array(losses_test) * 2, alpha=0.2,
                               color=color)
            for axes in ax:
                axes.set_yscale('log')
                axes.set_ylim([5e-4, 0.1])
                axes.grid()
                axes.set_xlabel(r'$\tau$')
            ax[0].set_ylabel(r'Cost/Risk')
            solid_line = mpl.lines.Line2D([0], [0], color='black', linestyle='solid', linewidth=2)
            dotted_line = mpl.lines.Line2D([0], [0], color='black', linestyle='dotted', linewidth=2)
            dashed_line = mpl.lines.Line2D([0], [0], color='black', linestyle='dashed', linewidth=2)
            labels = [r'$C_{\mathcal{D}_{\mathrm{Test}}}$', r'$R_{\mathcal{P}_{\mathrm{Haar}}}$',
                      r'$2C_{\mathcal{D}_{\mathrm{Test}}}$', ]
            leg2 = plt.legend([dotted_line, solid_line, dashed_line], labels, loc='upper right', prop={"size": 20})
            ax[0].add_artist(leg2)
            plt.tight_layout()
            fig.savefig(f'./figures/heisenberg_generalization.pdf')
        plt.show()


def figure_5(Lx, Ly):
    config = {
        # MODEL
        'Lx': Lx,
        'Ly': Ly,
        'hamiltonian': 'heisenberg_2d',
        't': 0.1,
        'boundary_condition': (True, False),
        'ctg': True,
        # TEBD
        'test_size': 100,  # number of states in test ensemble
        'tebd_test_steps': 20,  # number of steps in TEBD
        'max_bond': None,  # maximum bond dimension in tebd
        'tebd_cutoff': 1e-9,  # maximum bond dimension in tebd
        # 'ctg': True,
        # TRAINING
        'circuit_name': 'brickwall',
        'circuit_translation': True,  # translation invariant circuit
        'num_steps': 2000,  # maximum number of optimization steps
        'num_samples': 4,  # number of training samples
        'training_states': 'product',
        # STRATEGY`
        'training_strategy': 'hotstart',
        # HOTSTART
        'depth_min': 1,  # maximum circuit depth
        'depth_max': 5,  # maximum circuit depth
        'depth_step': 1,  # increase depth per step
        'trotter_start': False,  # increase depth per step
        # OPTIMIZATION
        'learning_rate': 0.001,
        'learning_rate_schedule': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5),
        # META
        'TRAIN': True,
        'TEST': True,
        'TEST_UNITARY': False,
        'PLOT': False,
        'PRINT': False,
        'SHOW': False,
        'SEED': 0
    }
    config['max_factor'] = 20
    config['start_depth'] = 4
    config['tebd_cutoff_circuit'] = 1e-5
    config['tebd_granularity'] = 100
    config['method'] = 'stacked'
    inds = [30, 31, 32]
    binary = ['0' if i not in inds else '1' for i in range(Lx * Ly)]
    psi0 = qtn.MPS_computational_state(binary)
    config['initial_state'] = psi0

    if GET_DATA:
        main(config)
        main_evolution(config)
    if PLOT:
        config['GET_PATH'] = True
        save_path = main_evolution(config)
        t = config['t']
        L = Lx * Ly
        magnetization_per_t_tebd = []
        magnetization_per_t_var = []
        corr_per_t_tebd = []
        corr_per_t_var = []
        spsm_per_t_tebd = []
        spsm_per_t_var = []

        times = [0.0] + [t * (i + 1) for i in range(config['max_factor'])]
        sz = quimb.spin_operator('Z')
        sp = quimb.spin_operator('+')
        sm = quimb.spin_operator('-')
        figpath = save_path + 'figures/'
        if not os.path.exists(figpath):
            os.makedirs(figpath)
        if not os.path.exists(save_path + f'data_magnetization_tebd_{config["max_factor"]}.npy'):
            dic_2d_1d = snake_index(Lx, Ly)
            dic_1d_2d = dict(zip(dic_2d_1d.values(), dic_2d_1d.keys()))
            save_path_t_i = save_path + f't_{t:1.3f}/depth_{config["start_depth"]}/'
            with open(save_path_t_i + 'psi0.pickle', 'rb') as file:
                psi0 = pickle.load(file)[0]
                for _i in range(L):
                    tn = psi0[_i]
                    tn.reindex({f"k{dic_1d_2d[_i]}": f"k{_i}"}, inplace=True)

            mz_j_tebd = np.zeros((Lx, Ly))
            mz_j_var = np.zeros((Lx, Ly))
            corr_j_tebd = np.zeros((Lx, Ly))
            corr_j_var = np.zeros((Lx, Ly))
            for lx in range(Lx):
                for ly in range(Ly):
                    idy = dic_2d_1d[(lx, ly)]
                    mz_j_tebd[lx, ly] = psi0.magnetization(idy, direction='Z')
                    mz_j_var[lx, ly] = mz_j_tebd[lx, ly].copy()
                    corr_j_tebd[lx, ly] = psi0.correlation(sz, 4, idy)
                    corr_j_var[lx, ly] = corr_j_tebd[lx, ly].copy()
            spsm_tebd = np.zeros((L, L), dtype=complex)
            spsm_var = np.zeros((L, L), dtype=complex)
            for idx in np.stack(np.tril_indices(L, k=-1)).T:
                spsm_tebd[idx[0], idx[1]] = psi0.correlation(sp, idx[0], idx[1], B=sm)
                spsm_var[idx[0], idx[1]] = psi0.correlation(sp, idx[0], idx[1], B=sm)
            spsm_tebd = spsm_tebd + spsm_tebd.T - np.diag(np.diag(spsm_tebd))
            spsm_var = spsm_var + spsm_var.T - np.diag(np.diag(spsm_var))
            spsm_per_t_tebd.append(spsm_tebd.copy())
            spsm_per_t_var.append(spsm_var.copy())
            magnetization_per_t_tebd.append(mz_j_tebd)
            magnetization_per_t_var.append(mz_j_var)
            corr_per_t_tebd.append(mz_j_tebd)
            corr_per_t_var.append(mz_j_var)
            for i in range(config['max_factor']):
                t_i = t * (i + 1)
                save_path_t_i = save_path + f't_{t_i:1.3f}/depth_{config["start_depth"]}/'
                with open(save_path_t_i + 'psit.pickle', 'rb') as file:
                    psit = pickle.load(file)[0]
                    for _i in range(L):
                        tn = psit[_i]
                        tn.reindex({f"k{dic_1d_2d[_i]}": f"k{_i}"}, inplace=True)
                with open(save_path_t_i + 'psit_var.pickle', 'rb') as file:
                    psit_var = pickle.load(file)[0]
                    for _i in range(L):
                        tn = psit_var[_i]
                        tn.reindex({f"k{dic_1d_2d[_i]}": f"k{_i}"}, inplace=True)
                mz_j_tebd = np.zeros((Lx, Ly))
                mz_j_var = np.zeros((Lx, Ly))
                for lx in range(Lx):
                    for ly in range(Ly):
                        idy = dic_2d_1d[(lx, ly)]
                        mz_j_tebd[lx, ly] = psit.magnetization(idy, direction='Z')
                        mz_j_var[lx, ly] = psit_var.magnetization(idy, direction='Z')
                        corr_j_tebd[lx, ly] = psit.correlation(sz, 4, idy)
                        corr_j_var[lx, ly] = psit_var.correlation(sz, 4, idy)
                spsm_tebd = np.zeros((L, L), dtype=complex)
                spsm_var = np.zeros((L, L), dtype=complex)
                for idx in np.stack(np.tril_indices(L, k=-1)).T:
                    spsm_tebd[idx[0], idx[1]] = psit.correlation(sp, idx[0], idx[1], B=sm)
                    spsm_var[idx[0], idx[1]] = psit_var.correlation(sp, idx[0], idx[1], B=sm)
                spsm_tebd = spsm_tebd + spsm_tebd.T - np.diag(np.diag(spsm_tebd))
                spsm_var = spsm_var + spsm_var.T - np.diag(np.diag(spsm_var))
                spsm_per_t_tebd.append(spsm_tebd.copy())
                spsm_per_t_var.append(spsm_var.copy())
                magnetization_per_t_tebd.append(mz_j_tebd)
                magnetization_per_t_var.append(mz_j_var)
                corr_per_t_tebd.append(corr_j_tebd)
                corr_per_t_var.append(corr_j_var)
            np.save(save_path + f'data_magnetization_tebd_{config["max_factor"]}', np.real(magnetization_per_t_tebd))
            np.save(save_path + f'data_magnetization_var_{config["max_factor"]}', np.real(magnetization_per_t_var))
            np.save(save_path + f'data_corr_tebd_{config["max_factor"]}', np.real(corr_per_t_tebd))
            np.save(save_path + f'data_corr_var_{config["max_factor"]}', np.real(corr_per_t_var))
            np.save(save_path + f'spsm_{config["max_factor"]}_tebd', np.real(spsm_per_t_tebd))
            np.save(save_path + f'spsm_{config["max_factor"]}_var', np.real(spsm_per_t_var))

        data_tebd = np.load(save_path + f'data_magnetization_tebd_{config["max_factor"]}.npy')
        data_var = np.load(save_path + f'data_magnetization_var_{config["max_factor"]}.npy')
        data_spsm_tebd = np.load(save_path + f'spsm_{config["max_factor"]}_tebd.npy')
        data_spsm_var = np.load(save_path + f'spsm_{config["max_factor"]}_var.npy')

        # plot the magnetization
        time_idx = [0, 5, 10, config["max_factor"]]

        dic_2d_1d = snake_index(Lx, Ly)
        dic_1d_2d = dict(zip(dic_2d_1d.values(), dic_2d_1d.keys()))
        with mpl.rc_context({'font.size': 18, 'font.family': 'serif', "text.usetex": True}):
            fig, axs = plt.subplots(2, len(time_idx))
            fig.set_size_inches(1 * len(time_idx), 5)
            plt.set_cmap('coolwarm')
            v_max = max(np.max(data_tebd), np.max(data_var))
            v_min = min(np.min(data_tebd), np.max(data_var))
            for i, k in enumerate(time_idx):
                data = [data_tebd[k].T, data_var[k].T]
                for j in range(2):
                    cf1 = axs[j, i].pcolormesh(np.arange(1, Lx + 1), np.arange(1, Ly + 1), data[j], vmin=v_min,
                                               vmax=v_max, edgecolors='black', linewidth=0.5)
                    if j == 0:
                        axs[j, i].set_title(f'T={times[k]:1.2f}')
                        axs[j, i].title.set_size(10)
                    if i == 0:
                        if j == 0:
                            axs[j, i].set_ylabel('$y$ (TEBD)')
                        else:
                            axs[j, i].set_ylabel('$y$ (Variational)')
                    if j == 1:
                        axs[j, i].set_xlabel('$x$')
                    if i > 0:
                        cf1.axes.set_yticks(())
                        cf1.axes.tick_params(axis="x", which='both', top=False, right=False)
                        cf1.axes.tick_params(axis="y", which='both', top=False, right=False, left=False)
                    else:
                        cf1.axes.set_yticks([])  # list(range(1, Ly+1))[::3])
                    cf1.axes.set_xticks([])  # list(range(1,Lx+1)))
            cbar_ax = fig.add_axes([0.8, 0.15, 0.05, 0.63])
            fig.colorbar(cf1, cax=cbar_ax)
            cbar_ax.set_title(r'$\langle \sigma^z \rangle$')
            fig.subplots_adjust(wspace=0.2, bottom=0.1, right=0.7)
            fig.savefig('./figures/lightcone_2d.pdf')
        with mpl.rc_context({'font.size': 18, 'font.family': 'serif', "text.usetex": True}):

            k_corrs_per_t_tebd = []
            k_corrs_per_t_var = []
            for i, k in enumerate(time_idx):
                xs = np.array(list(range(-(Lx - 1) // 2, 0)) + list(range(0, (Lx + 1) // 2))) / Lx * np.pi * 2
                ys = np.array(list(range(-(Ly - 1) // 2, 0)) + list(range(0, (Ly + 1) // 2))) / Ly * np.pi * 2
                k_corrs_tebd = np.zeros((len(xs), len(ys)), dtype=complex)
                k_corrs_var = np.zeros((len(xs), len(ys)), dtype=complex)
                for idx in np.stack(np.tril_indices(L, k=-1)).T:
                    assert Lx == 3, "Hacky way to do closed boundaries right now...."
                    ri = np.array(dic_1d_2d[idx[0]])
                    rj = np.array(dic_1d_2d[idx[1]])
                    r = ri - rj
                    if ri[0] == rj[0]:
                        r[0] = 0.
                    elif ri[0] > rj[0]:
                        r[0] = 1.
                    else:
                        r[0] = -1.

                    k_corrs_tebd += np.exp(1j * (r[0] * xs[:, np.newaxis] + r[1] * ys[np.newaxis, :])) * \
                                    data_spsm_tebd[k, idx[0], idx[1]]
                    k_corrs_var += np.exp(1j * (r[0] * xs[:, np.newaxis] + r[1] * ys[np.newaxis, :])) * \
                                   data_spsm_var[k, idx[0], idx[1]]
                k_corrs_per_t_tebd.append(k_corrs_tebd)
                k_corrs_per_t_var.append(k_corrs_var)
            k_corrs_per_t_tebd = np.stack(k_corrs_per_t_tebd)
            k_corrs_per_t_var = np.stack(k_corrs_per_t_var)

            fig, axs = plt.subplots(2, len(time_idx))
            fig.set_size_inches(1 * len(time_idx), 5)
            v_max = max(np.max(k_corrs_per_t_tebd.real), np.max(k_corrs_per_t_tebd.real))
            v_min = min(np.min(k_corrs_per_t_tebd.real), np.max(k_corrs_per_t_tebd.real))
            for i, k in enumerate(time_idx):
                xs = np.array(list(range(-(Lx - 1) // 2, 0)) + list(range(0, (Lx + 1) // 2))) / Lx * np.pi * 2
                ys = np.array(list(range(-(Ly - 1) // 2, 0)) + list(range(0, (Ly + 1) // 2))) / Ly * np.pi * 2
                mesh = np.meshgrid(xs, ys)
                data = [k_corrs_per_t_var[i].real.T, k_corrs_per_t_tebd[i].real.T]
                for j in range(2):
                    cf = axs[j, i].pcolormesh(mesh[0], mesh[1], data[j], cmap=plt.get_cmap('viridis'), vmin=v_min,
                                              vmax=v_max, edgecolors='black', linewidth=0.5)
                    if j == 0:
                        axs[j, i].set_title(f'T={times[k]:1.2f}')
                        axs[j, i].title.set_size(10)
                    if i == 0:
                        if j == 0:
                            axs[j, i].set_ylabel(r'$k_y$ (TEBD)')
                        else:
                            axs[j, i].set_ylabel(r'$k_y$ (Variational)')
                    if j == 1:
                        axs[j, i].set_xlabel(r'$k_x$')
                    if i > 0:
                        cf.axes.set_yticks(())
                        cf.axes.tick_params(axis="x", which='both', top=False, right=False)
                        cf.axes.tick_params(axis="y", which='both', top=False, right=False, left=False)
                    else:
                        cf.axes.set_yticks([])
                    cf.axes.set_xticks([])

            cbar_ax = fig.add_axes([0.8, 0.15, 0.05, 0.63])
            fig.colorbar(cf, cax=cbar_ax)
            cbar_ax.set_title(r'$S_k$')
            fig.subplots_adjust(wspace=0.2, bottom=0.1, right=0.7)
            fig.savefig('./figures/momentum_2d.pdf')
            plt.show()


def figure_6a():
    raise NotImplementedError


def figure_6b():
    raise NotImplementedError


def figure_6c():
    raise NotImplementedError


def figure_7(L):
    config = {
        # MODEL
        'L': L,
        'hamiltonian': 'heisenberg',
        't': 0.1,
        # TEBD
        'test_size': 100,  # number of states in test ensemble
        'tebd_test_steps': 20,  # number of steps in TEBD
        'max_bond': None,  # maximum bond dimension in tebd
        'tebd_cutoff': 1e-9,  # maximum bond dimension in tebd
        # 'ctg': True,
        # TRAINING
        'circuit_name': 'brickwall',
        'circuit_translation': False,  # translation invariant circuit
        'num_steps': 2000,  # maximum number of optimization steps
        'num_samples': 16,  # number of training samples
        'training_states': 'product',
        'num_particles': 10,
        # STRATEGY`
        'training_strategy': 'hotstart',
        # HOTSTART
        'depth_max': 4,  # maximum circuit depth
        'depth_min': 2,  # increase depth per step
        'depth_step': 1,  # increase depth per step
        'trotter_start': False,  # increase depth per step
        # OPTIMIZATION
        'learning_rate': 0.001,
        'learning_rate_schedule': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5),
        # META
        'TRAIN': True,
        'TEST': True,
        'TEST_UNITARY': False,
        'PRINT': False,
        'PLOT': False,
        'SHOW': False,
        'SEED': 0
    }
    config['max_factor'] = 10
    config['start_depth'] = 4
    config['method'] = 'stacked'
    config['tebd_cutoff_circuit'] = 1e-7,
    psi0 = random_U1_state(L, 3, config['num_particles'], 1000)
    config['initial_state'] = psi0
    if GET_DATA:
        main(config)
        main_evolution(config)
        config['training_states'] = 'u1'
        main(config)
        main_evolution(config)
    if PLOT:
        labels = ['Product', '$U(1)$']
        MS = 10
        with mpl.rc_context({'font.size': 18, 'font.family': 'serif', "text.usetex": True}):
            fig, axs = plt.subplots(1, 1)
            fig.set_size_inches(5, 5)
            cmap = plt.get_cmap('tab20')
            colors = [cmap(0.2), cmap(0.3)]
            for state_i, training_state in enumerate(['product', 'u1']):
                config['training_states'] = training_state
                T_list, bds, infids = main_evolution(config)
                axs.plot(T_list, np.array(infids), label=labels[state_i], color=colors[state_i], markersize=MS,
                         marker='.')
            axs.legend()
            axs.set_yscale('log')
            axs.set_xscale('log')
            axs.grid()
            axs.legend(prop={"size": 15})
            axs.set_xlabel(r'Time $T$')
            axs.set_ylabel(r'$C_{\mathcal{D}_{\mathrm{State}}}$')
            plt.tight_layout()
            fig.savefig(f'./figures/u1_fidelity.pdf')
            plt.show()


def figure_8():
    if GET_DATA:
        config = {
            # MODEL
            'L': 80,
            'hamiltonian': 'ising',
            't': 1.0,
            # TEBD
            'test_size': 100,  # number of states in test ensemble
            'tebd_test_steps': 20,  # number of steps in TEBD
            'max_bond': None,  # maximum bond dimension in tebd
            'tebd_cutoff': 1e-9,  # maximum bond dimension in tebd
            # TRAINING
            'circuit_name': 'brickwall',
            'circuit_translation': True,  # translation invariant circuit
            'num_steps': 2000,  # maximum number of optimization steps
            'num_samples': 16,  # number of training samples
            'training_states': 'product',
            # STRATEGY`
            'training_strategy': 'hotstart',
            # 'training_strategy': 'double_space',
            # 'training_strategy': 'double_time',
            # HOTSTART
            'depth_max': 8,  # maximum circuit depth
            'depth_min': 1,  # increase depth per step
            'depth_step': 1,  # increase depth per step
            'trotter_start': False,  # increase depth per step
            'trotter_start_order': 1,  # increase depth per step
            # OPTIMIZATION
            'learning_rate': 0.001,
            'learning_rate_schedule': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5),
            # META
            'TRAIN': True,
            'TEST': True,
            'TEST_UNITARY': False,
            'PRINT': False,
            'PLOT': False,
            'SHOW': False,
            'SEED': 0
        }
        main(config)
        config = {
            # MODEL
            'L': 80,
            'hamiltonian': 'ising',
            't': 1.0,
            # TEBD
            'test_size': 100,  # number of states in test ensemble
            'tebd_test_steps': 20,  # number of steps in TEBD
            'max_bond': None,  # maximum bond dimension in tebd
            'tebd_cutoff': 1e-9,  # maximum bond dimension in tebd
            # TRAINING
            'circuit_name': 'brickwall',
            'circuit_translation': True,  # translation invariant circuit
            'num_steps': 2000,  # maximum number of optimization steps
            'num_samples': 16,  # number of training samples
            'training_states': 'product',
            # STRATEGY`
            'training_strategy': 'hotstart',
            # 'training_strategy': 'double_space',
            # 'training_strategy': 'double_time',
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
            'PRINT': False,
            'SHOW': False,
            'SEED': 0
        }
        main(config)
        config = {
            # MODEL
            'L': 10,
            'hamiltonian': 'ising',
            't': 1.0,
            # TEBD
            'test_size': 100,  # number of states in test ensemble
            'tebd_test_steps': 20,  # number of steps in TEBD
            'max_bond': None,  # maximum bond dimension in tebd
            'tebd_cutoff': 1e-9,  # maximum bond dimension in tebd
            # TRAINING
            'circuit_name': 'brickwall',
            'circuit_translation': True,  # translation invariant circuit
            'num_steps': 2000,  # maximum number of optimization steps
            'num_samples': 16,  # number of training samples
            'training_states': 'product',
            # STRATEGY`
            # 'training_strategy': 'hotstart',
            'training_strategy': 'double_space',
            # 'training_strategy': 'double_time',
            # TIME/SPACE DOUBLING
            'max_factor': 4,  # maximum factor f-> 2**f of intitial time
            'depth': 8,  # initial depth, will double each time
            # OPTIMIZATION
            'learning_rate': 0.001,
            'learning_rate_schedule': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5),
            # META
            'TRAIN': True,
            'TEST': True,
            'TEST_UNITARY': False,
            'PLOT': False,
            'PRINT': False,
            'SHOW': False,
            'SEED': 0
        }
        main(config)
        config = {
            # MODEL
            'L': 80,
            'hamiltonian': 'ising',
            't': 0.25,
            # TEBD
            'test_size': 100,  # number of states in test ensemble
            'tebd_test_steps': 20,  # number of steps in TEBD
            'tebd_granularity': 20,  # number of steps in TEBD
            'max_bond': None,  # maximum bond dimension in tebd
            'tebd_cutoff': 1e-9,  # maximum bond dimension in tebd
            # TRAINING
            'circuit_name': 'brickwall',
            'circuit_translation': True,  # translation invariant circuit
            'num_steps': 2000,  # maximum number of optimization steps
            'num_samples': 16,  # number of training samples
            'training_states': 'product',
            # STRATEGY`
            'training_strategy': 'double_time',
            # HOTSTART
            # TIME/SPACE DOUBLING
            'max_factor': 3,  # maximum factor f-> 2**f of intitial time
            'initial_depth': 2,  # initial depth, will double each time
            # OPTIMIZATION
            'learning_rate': 0.001,
            'learning_rate_schedule': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5),
            # META
            'TRAIN': True,
            'TEST': True,
            'TEST_UNITARY': False,
            'PLOT': False,
            'PRINT': False,
            'SHOW': False,
            'SEED': 0
        }
        main(config)
        config = {
            # MODEL
            'L': 80,
            'hamiltonian': 'ising',
            't': 1.0,
            # TEBD
            'test_size': 100,  # number of states in test ensemble
            'tebd_test_steps': 20,  # number of steps in TEBD
            'max_bond': None,  # maximum bond dimension in tebd
            'tebd_cutoff': 1e-9,  # maximum bond dimension in tebd
            # TRAINING
            'circuit_name': 'brickwall',
            'circuit_translation': True,  # translation invariant circuit
            'num_steps': 2000,  # maximum number of optimization steps
            'num_samples': 16,  # number of training samples
            'training_states': 'product',
            # STRATEGY`
            'training_strategy': 'hotstart',
            # 'training_strategy': 'double_space',
            # 'training_strategy': 'double_time',
            # HOTSTART
            'depth_max': 9,  # maximum circuit depth
            'depth_min': 2,  # increase depth per step
            'depth_step': 1,  # increase depth per step
            'trotter_start': True,  # increase depth per step
            'trotter_start_order': 1,  # increase depth per step
            # OPTIMIZATION
            'learning_rate': 0.001,
            'learning_rate_schedule': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5),
            # META
            'TRAIN': True,
            'TEST': True,
            'TEST_UNITARY': False,
            'PLOT': False,
            'PRINT': False,
            'SHOW': False,
            'SEED': 0
        }
        main(config)

    if PLOT:
        data = {}
        missing_data = 0
        data_total = 0
        pre_path = './data/UNITARY_COMPILATION/ising_g_1.00/'
        for depth in range(1, 9):
            path = pre_path + f'L_80/hotstart_dstep_1/t_1.000/Nsteps_2000_brickwall_translation/SEED_0/Ns_16_product_bd_max_None/depth_{depth}/'
            data_total += 1
            try:
                data[f'hotstart_{depth}'] = np.load(path + 'test_loss_100.npy')
            except FileNotFoundError:
                data[f'hotstart_{depth}'] = np.nan
                missing_data += 1
        for depth in range(5, 9):
            path = pre_path + f'L_80/hotstart_dstep_1/t_1.000/Nsteps_2000_brickwall_translation_trotter_init_2/SEED_0/Ns_16_product_bd_max_None/depth_{depth}/'
            data_total += 1
            try:
                data[f'hotstart_trotter_2{depth}'] = np.load(path + 'test_loss_100.npy')
            except FileNotFoundError:
                data[f'hotstart_trotter_2{depth}'] = np.nan
                missing_data += 1
        for depth in range(2, 9):
            path = pre_path + f'L_80/hotstart_dstep_1/t_1.000/Nsteps_2000_brickwall_translation_trotter_init_1/SEED_0/Ns_16_product_bd_max_None/depth_{depth}/'
            data_total += 1
            try:
                data[f'hotstart_trotter_1{depth}'] = np.load(path + 'test_loss_100.npy')
            except FileNotFoundError:
                data[f'hotstart_trotter_1{depth}'] = np.nan
                missing_data += 1
        for factor in range(3):
            t = 0.25
            t_i = t * 2 ** factor
            depth = 2 * 2 ** (factor)
            path = pre_path + f'L_80/double_time_initial_depth_2/Nsteps_2000_brickwall_translation/SEED_0/t_start_0.250/Ns_16_product_bd_max_None/t_{t_i:1.3f}/depth_{depth}/'
            data_total += 1
            try:
                data[f'double_time_{depth}'] = np.load(path + 'test_loss_100.npy')
            except FileNotFoundError:
                data[f'double_time_{depth}'] = np.nan
                missing_data += 1
        for factor in range(4):
            L = 10
            L_i = L * 2 ** factor
            path = pre_path + f'L_10/double_space_depth_8/Nsteps_2000_brickwall_translation/SEED_0/t_1.000/L_start_10/Ns_16_product_bd_max_None/L_{L_i}/depth_8/'
            data_total += 1
            try:
                data[f'double_space_{L_i}'] = np.load(path + 'test_loss_100.npy')
            except FileNotFoundError:
                data[f'double_space_{L_i}'] = np.nan
                missing_data += 1
        print(f"Missing data {data_total - missing_data}/{data_total}")
        MS = 10
        with mpl.rc_context({'font.size': 18, 'font.family': 'serif', "text.usetex": True}):
            fig, ax = plt.subplots(1, 1)
            ax = [ax, ]
            fig.set_size_inches(6, 5)
            cmap = plt.get_cmap('Set1')
            colors = cmap(np.linspace(0., 0.5, 5))
            data_1 = []
            data_2a = []
            data_2b = []
            data_3 = []
            data_4 = []
            for depth in range(1, 9):
                data_1.append(data[f'hotstart_{depth}'])
            ax[0].plot(list(range(1, 9)), data_1, color=colors[0], label=rf"No warm start", marker='.', markersize=MS)
            for depth in range(2, 9):
                data_2a.append(data[f'hotstart_trotter_1{depth}'])
            ax[0].plot(list(range(2, 9)), np.array(data_2a) * 0.9, color=colors[1], label=rf"$p=1$ Trotter", marker='.',
                       markersize=MS)
            for depth in range(5, 9):
                data_2b.append(data[f'hotstart_trotter_2{depth}'])
            ax[0].plot(list(range(5, 9)), data_2b, color=colors[2], label=rf"$p=2$ Trotter", marker='.', markersize=MS)

            depths = []
            for factor in range(3):
                depth = 2 * 2 ** (factor)
                depths.append(depth)
                data_3.append(data[f'double_time_{depth}'])
            x = depths
            y = data_3
            ax[0].plot(x, y, color=colors[3], label=rf"Double time", marker='.', markersize=MS)
            changes = [(1.1, 4), (1.1, 4), (0.95, 0.1)]
            for i, txt in enumerate(range(3)):
                t_i = 0.25 * 2 ** i
                ax[0].annotate(f"t={t_i:1.3f}",
                               xy=(x[i], y[i]), ha="center",
                               xytext=(x[i] * changes[i][0], y[i] * changes[i][1]),
                               arrowprops=dict(arrowstyle="-"))
            for factor in range(4):
                L = 10
                L_i = L * 2 ** factor
                data_4.append(data[f'double_space_{L_i}'])
            x = [8] * 4
            y = data_4
            ax[0].scatter(x, y, color=colors[4], marker='.', s=100)
            ax[0].plot(x, y, color=colors[4], label=rf"Double space", marker='.', markersize=MS)
            changes = [(0.8, 1.5), (0.8, 1.5), (1.05, 10), (1.05, 1)]
            for i, txt in enumerate(range(4)):
                L_i = 10 * 2 ** i
                ax[0].annotate(f"$n$={L_i}",
                               xy=(x[i], y[i]), va='center',
                               xytext=(x[i] * changes[i][0], y[i] * changes[i][1]),
                               arrowprops=dict(arrowstyle="-"))
            for axes in ax:
                axes.set_yscale('log')
                axes.set_ylim([1e-7, 10.])
                axes.grid()
                axes.set_xlabel(r'$\tau$')
            ax[0].set_ylabel(r'$C_{\mathcal{D}_{\mathrm{Test}}}$')
            ax[0].legend(loc='lower left', prop={"size": 13})
            ax[0].set_xlim([1, 10])
            plt.tight_layout()
            fig.savefig(f'./figures/appendix_warm_start.pdf')
            plt.show()


if __name__ == '__main__':
    GET_DATA = True
    PLOT = True
    figure_3a(L=80)
    figure_3b(L=60)
    figure_3c()
    figure_4a(L=40)
    figure_4b(L=40)
    figure_4c(L=40)
    figure_5(Lx=3, Ly=21)
    figure_7(L=20)
    figure_8()
