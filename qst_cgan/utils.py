import numpy as np
import os

import h5py

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt
import os
import pickle

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from qutip.wigner import qfunc
from qutip.visualization import plot_fock_distribution
from qutip import Qobj
from matplotlib import cm

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {# 'backend': 'ps',
          'axes.labelsize': 8,
          'font.size': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'axes.labelpad': 1,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
figpath = "figures/"

# Adopted from the SciPy Cookbook.
def _blob(x, y, w, w_max, area, cmap=None, ax=None):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])

    ax.fill(xcorners, ycorners,
             color=cmap(int((w + w_max) * 256 / (2 * w_max))))




# Adopted from the SciPy Cookbook.
def hinton(rho, xlabels=None, ylabels=None, title=None, fig=None, ax=None, cmap=None,
           label_top=True):
    """Draws a Hinton diagram for visualizing a density matrix or superoperator.

    Parameters
    ----------
    rho : qobj
        Input density matrix or superoperator.

    xlabels : list of strings or False
        list of x labels

    ylabels : list of strings or False
        list of y labels

    title : string
        title of the plot (optional)

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.

    cmap : a matplotlib colormap instance
        Color map to use when plotting.

    label_top : bool
        If True, x-axis labels will be placed on top, otherwise
        they will appear below the plot.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not a quantum object.

    """

    # Apply default colormaps.
    # TODO: abstract this away into something that makes default
    #       colormaps.
    cmap = cm.RdBu

    # Extract plotting data W from the input.
    W = rho.full()
    ax.axis('equal')
    ax.set_frame_on(False)

    height, width = W.shape

    w_max = 1.25 * max(abs(np.diag(np.matrix(W))))
    if w_max <= 0.0:
        w_max = 1.0

    ax.fill(np.array([0, width, width, 0]), np.array([0, 0, height, height]),
            color=cmap(128))
    for x in range(width):
        for y in range(height):
            _x = x + 1
            _y = y + 1
            if np.real(W[x, y]) > 0.0:
                _blob(_x - 0.5, height - _y + 0.5, abs(W[x,
                      y]), w_max, min(1, abs(W[x, y]) / w_max), cmap=cmap)
            else:
                _blob(_x - 0.5, height - _y + 0.5, -abs(W[
                      x, y]), w_max, min(1, abs(W[x, y]) / w_max), cmap=cmap)


def read_file(path):
    """
    Reads a file
    """
    flist = None
    with open(path, 'rb') as f:
        flist = pickle.load(f)
    return flist


def save_states(states_list, path):
    """
    Saves fidelities and appends to the file if it exists already.

    Parameters
    ----------
    states_list : list
        A list of states.

    path : str
        A path to a file to save the fidelities. If the file exists
    """
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            slist = pickle.load(f)
        slist.append(states_list)
    else:
        slist = []
        slist.append(states_list)

    with open(path, 'wb') as f:
        pickle.dump(slist, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_fidelities(fidelity_list, path):
        """
        Saves fidelities and appends to the file if it exists already.
        
        Parameters
        ----------
        fidelity_list : list
            A list of fidelities.
            
        path : str
            A path to a file to save the fidelities. If the file exists
        """
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                flist = pickle.load(f)
            flist.append(fidelity_list)
        else:
            flist = []
            flist.append(fidelity_list)
            
        with open(path, 'wb') as f:
            pickle.dump(flist, f, protocol=pickle.HIGHEST_PROTOCOL)


class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc.
    
    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype
    
    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)
        
    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """
    def __init__(self, datapath, fields,
                 compression="gzip", chunks=True):
        
        self.datapath = datapath
        self.fields = fields
        self.field_names = fields.keys()

        if os.path.isfile(datapath):
            self.datasets = {}
            with h5py.File(self.datapath, mode='r') as f:
                temp_name = ''
                for name in self.field_names:
                    self.datasets[name] = f[name]
                    temp_name = name
                self.i = f[temp_name].shape[0]
            print("File exists with {} entries. Append mode".format(self.i))
        else:
            self.i = 0
            self.datasets = {}
            with h5py.File(self.datapath, mode='w') as h5f:
                for name in self.field_names:
                    shape = fields[name][0]
                    dtype = fields[name][1]

                    self.datasets[name] = h5f.create_dataset(
                        name,
                        shape=(0, ) + shape,
                        maxshape=(None, ) + shape,
                        dtype = dtype,
                        compression = compression,
                        chunks=chunks)
    
    def append(self, value_dict):
        with h5py.File(self.datapath, mode='a') as h5f:
            for name in value_dict:
                shape = self.fields[name][0]
                
                dset = h5f[name]
                dset.resize((self.i + 1, ) + shape)
                dset[self.i] = [value_dict[name]]
            self.i += 1
            h5f.flush()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          fig = None,
                          ax=None,
                          cax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
#     # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    if fig == None:
        fig, ax = plt.subplots(figsize=(7, 4))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    if cax == None:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    else:
        cbar = fig.colorbar(im, cax=cax, pad=0.03, ticks=[0, 0.5, 1])
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if (cm[i, j]  == 0.):
                l = str(0)
            else:
                l = format(cm[i, j], fmt)
            if float(l) < 1e-3:    
                l = str(0)

            if float(l) == 1.:    
                l = str(1)
            ax.text(j, i, l, fontsize=6,
                    ha="center", va="center",
                    color="w" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax











def plot_husimi_data(rho, title=""):
    """
    """
    # params = {# 'backend': 'ps',
    #       'axes.labelsize': 6,
    #       'font.size': 6,
    #       'xtick.labelsize': 6,
    #       'ytick.labelsize': 6,
    #       'axes.labelpad': 1,
    #       'text.usetex': False,
    #       'figure.figsize': fig_size}
    # plt.rcParams.update(params)
    

#     rho_tf = dm_to_tf([rho])
#     data = batched_expect(ops_batch_husimi, rho_tf)
#     x = data.numpy().reshape((len(xvec), len(xvec)))
    xvec = np.linspace(-5, 5, 32)
    yvec = np.linspace(-5, 5, 32)
    x = qfunc(rho, xvec*np.sqrt(2), yvec*np.sqrt(2))

    fig, ax = plt.subplots(1, 1, figsize=(fig_width/3.5, fig_width/3.5))
    
    im = ax.pcolor(xvec, yvec, x/np.max(x), cmap="hot", vmin=0., vmax=1)
    ax.set_aspect("equal")
    ax.set_yticks([-5, 0, 5])
    ax.set_xlabel(r"Re$(\beta)$", labelpad=0)
    ax.set_ylabel(r"Im$(\beta)$", labelpad=-5)
    cbar = plt.colorbar(im, fraction=0.0455, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_title(title, y=0.93)
    cbar.solids.set_edgecolor("face")
    
    return fig, ax


def plot_husimi_directly(x, title=""):
    """
    """
    # params = {# 'backend': 'ps',
    #       'axes.labelsize': 6,
    #       'font.size': 6,
    #       'xtick.labelsize': 6,
    #       'ytick.labelsize': 6,
    #       'axes.labelpad': 1,
    #       'text.usetex': False,
    #       'figure.figsize': fig_size}
    # plt.rcParams.update(params)
    

#     rho_tf = dm_to_tf([rho])
#     data = batched_expect(ops_batch_husimi, rho_tf)
#     x = data.numpy().reshape((len(xvec), len(xvec)))
    
    # x = qfunc(rho, xvec*np.sqrt(2), yvec*np.sqrt(2))
    xvec = np.linspace(-5, 5, 32)
    yvec = np.linspace(-5, 5, 32)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width/3.5, fig_width/3.5))
    
    im = ax.pcolor(xvec, yvec, x/np.max(x), cmap="hot", vmin=0., vmax=1)
    ax.set_aspect("equal")
    ax.set_yticks([-5, 0, 5])
    ax.set_xlabel(r"Re$(\beta)$", labelpad=0)
    ax.set_ylabel(r"Im$(\beta)$", labelpad=-5)
    cbar = plt.colorbar(im, fraction=0.0455, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_title(title, y=0.93)
    cbar.solids.set_edgecolor("face")


    return fig, ax



def plot_fock(rho, hinton_limit = 20, title="", ylim=0.5):
    """Draws a Hinton diagram for visualizing a density matrix or superoperator.

    Parameters
    ----------
    rho : qobj
        Input density matrix or superoperator.

    xlabels : list of strings or False
        list of x labels

    ylabels : list of strings or False
        list of y labels

    title : string
        title of the plot (optional)

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.

    cmap : a matplotlib colormap instance
        Color map to use when plotting.

    label_top : bool
        If True, x-axis labels will be placed on top, otherwise
        they will appear below the plot.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not a quantum object.

    """
    params = {# 'backend': 'ps',
          'axes.labelsize': 8,
          'font.size': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'axes.labelpad': -1,
          'text.usetex': False,
          'figure.figsize': fig_size}
    plt.rcParams.update(params)
    fig, ax = plot_fock_distribution(Qobj(rho.full()[:hinton_limit, :hinton_limit]),
                                     figsize=(fig_width/3.8, fig_width/3.8), )
    ax.set_title(title)
    ax.set_xlabel(r"|$n\rangle$", labelpad=-11, fontsize=8)
    ax.set_ylabel(r"p($n$)", labelpad=-16, fontsize=8)

    ax.set_ylim(0, ylim)
    ax.set_yticks([0, ylim/2, ylim])
    ax.set_yticklabels(["0", "", ylim])
    # ax.axis("off")
    return fig, ax



def plot_three_husimi(d1, d2, d3, title="", subtitles=None, cmap="hot"):
    """
    Plots three Husimi Q side by side
    """
    xvec = np.linspace(-5, 5, 32)
    yvec = np.linspace(-5, 5, 32)
    fig, ax = plt.subplots(1, 3, figsize=(fig_width, 0.35*fig_width))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    im = ax[0].pcolor(xvec, yvec, d1/np.max(d1),
                      norm=norm,
                      cmap=cmap)
    
    im = ax[1].pcolor(xvec, yvec, d2/np.max(d2),
                      norm=norm,
                      cmap=cmap)
    
    im = ax[2].pcolor(xvec, yvec, d3/np.max(d3),
                      norm=norm,
                      cmap=cmap)
    
    ax[0].set_yticklabels(["-5", "", "5"])

    for axis in ax:
        axis.set_xticks([-5, 0,  5])
        axis.set_yticks([-5, 0, 5])

        axis.set_xticklabels(["", "", ""])
        axis.set_yticklabels(["", "", ""])
        axis.set_xlabel(r"Re($\beta$)", labelpad=-9)    
        axis.set_aspect("equal")

    for i in range(0, 3):
        ax[i].set_xticklabels(["-5", "", "5"])
        ax[i].set_xticklabels(["-5", "", "5"])

        if subtitles != None:
            ax[i].set_title(subtitles[i])

    ax[0].set_ylabel(r"Im($\beta$)", labelpad=-12)
    ax[0].set_yticklabels(["-5", "", "5"])

    fig.subplots_adjust(right=0.85, wspace=0.01, hspace=-7.6)

    cax = fig.add_axes([0.8634, 0.2, 0.01, 0.602])
    fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cax.set_yticklabels(["0", 0.5, "1"])
    plt.subplots_adjust(wspace=0.25)

    if subtitles != None:
        if title:
            plt.suptitle(title)
    else:
        plt.suptitle(title, y=.98)
    return fig, ax




def plot_three_wigner(d1, d2, d3, title="", subtitles=None, cmap="RdBu"):
    """
    Plots three Husimi Q side by side
    """
    xvec = np.linspace(-5, 5, 100)
    yvec = np.linspace(-5, 5, 100)

    fig, ax = plt.subplots(1, 3, figsize=(fig_width, 0.35*fig_width))
    norm = matplotlib.colors.DivergingNorm(vmin=-np.max(d1), vcenter=0, vmax=np.max(d1))
    im = ax[0].pcolor(xvec, yvec, d1,
                      norm=norm,
                      cmap=cmap)
    

    norm = matplotlib.colors.DivergingNorm(vmin=-np.max(d2), vcenter=0, vmax=np.max(d2))
    im = ax[1].pcolor(xvec, yvec, d2,
                      norm=norm,
                      cmap=cmap)
    

    norm = matplotlib.colors.DivergingNorm(vmin=-np.max(d3), vcenter=0, vmax=np.max(d3))
    im = ax[2].pcolor(xvec, yvec, d3,
                      norm=norm,
                      cmap=cmap)
    
    ax[0].set_yticklabels(["-5", "", "5"])

    for axis in ax:
        axis.set_xticks([-5, 0,  5])
        axis.set_yticks([-5, 0, 5])

        axis.set_xticklabels(["", "", ""])
        axis.set_yticklabels(["", "", ""])
        axis.set_xlabel(r"Re($\beta$)", labelpad=-9)    
        axis.set_aspect("equal")

    for i in range(0, 3):
        ax[i].set_xticklabels(["-5", "", "5"])
        ax[i].set_xticklabels(["-5", "", "5"])

        if subtitles != None:
            ax[i].set_title(subtitles[i])

    ax[0].set_ylabel(r"Im($\beta$)", labelpad=-12)
    ax[0].set_yticklabels(["-5", "", "5"])

    fig.subplots_adjust(right=0.85, wspace=0.01, hspace=-7.6)

    cax = fig.add_axes([0.8634, 0.2, 0.01, 0.602])

    fig.colorbar(im, cax=cax, ticks=[-0.1, 0, 0.1])
    cax.set_yticklabels([-0.1, 0, 0.1])
    plt.subplots_adjust(wspace=0.25)

    if subtitles != None:
        if title:
            plt.suptitle(title)
    else:
        plt.suptitle(title, y=.98)
    return fig, ax