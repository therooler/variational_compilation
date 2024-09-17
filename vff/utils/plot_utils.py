import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import List, Union

try:
    matplotlib.use('TkAgg')
except ImportError:
    print("TkAgg not found...")
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', "text.usetex": False})
MARKERSIZE = 10


def plot_losses(depths: Union[List, np.ndarray],
                train_losses: np.ndarray,
                test_losses: np.ndarray = None,
                unitary_losses: np.ndarray = None,
                fig_path: str = None):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    axs.plot(depths, train_losses, marker='.', markersize=MARKERSIZE, label='Train')
    axs.plot(depths, test_losses, marker='.', markersize=MARKERSIZE, label='Test')
    axs.plot(depths, unitary_losses, marker='.', markersize=MARKERSIZE, label='Unitary')
    axs.grid()
    axs.set_yscale('log')
    axs.set_xlabel('Depth')
    axs.set_xticks(depths)
    axs.set_ylabel('Loss')
    axs.legend()
    if fig_path is not None:
        fig.savefig(fig_path + "losses.pdf")
    plt.tight_layout()
