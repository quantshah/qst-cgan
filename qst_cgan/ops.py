import tensorflow as tf

import numpy as np

from qutip import coherent_dm as qutip_coherent_dm
from qutip import thermal_dm as qutip_thermal_dm
from qutip import Qobj, fock, coherent, displace

from qutip.states import fock_dm as qutip_fock_dm
from qutip.states import thermal_dm as qutip_thermal_dm
from qutip.random_objects import rand_dm
from scipy.special import binom
from math import sqrt

import matplotlib.tri as tri
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
import matplotlib.pyplot as plt


def gaus2d(x=0, y=0, n0=1):
    return 1. / (np.pi * n0) * np.exp(-((x**2 + y**2.0)/n0))


class GaussianConv(tf.keras.layers.Layer):
    """
    Expectation layer that calculates expectation values for a set of operators on a batch of rhos.
    You can specify different sets of operators for each density matrix in the batch.
    """
    def __init__(self, kernel = None, **kwargs):
        super(GaussianConv, self).__init__(**kwargs)
        self.kernel = kernel[:, :, tf.newaxis, tf.newaxis]

    def call(self, x):
        """Expectation function call
        """
        return tf.nn.conv2d(x, self.kernel, strides=[1, 1, 1, 1], padding="SAME")


def expect(ops, rhos):
    """
    Calculates expectation values for a batch of density matrices
    for list of operators.
    
    Args:
        ops (`tf.Tensor`): a 3D tensor (N, hilbert_size, hilbert_size) of N
                                         measurement operators
        rhos (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size)
                            representing N density matrices        

    Returns:
        expectations (:class:`tensorflow.Tensor`): A 4D tensor (batch_size, N, 1)
                                                   giving expectation values for the
                                                   (N, N) grid of operators for
                                                   all the density matrices (batch_size).
    """
    products = tf.einsum("aij, jk->aik", ops, rhos)
    traces = tf.linalg.trace(products)
    expectations = tf.math.real(traces)
    return expectations


def random_alpha(radius, inner_radius=0):
    """
    Generates a random complex values within a circle
    
    Args:
        radius (float): Radius for the values
        inner_radius (float): Inner radius which defaults to 0.
    """
    radius = np.random.uniform(inner_radius, radius)
    phi = np.random.uniform(-np.pi, np.pi)
    return radius * np.exp(1j * phi)


def dm_to_tf(rhos):
    """
    Convert a list of qutip density matrices to TensorFlow 
    density matrices

    Args:
        rhos (list of `qutip.Qobj`): List of N qutip density matrices

    Returns:
        tf_dms (:class:`tensorflow.Tensor`): A 3D tensor (N, hilbert_size, hilbert_size) of N
                                         density matrices

    """
    tf_dms = tf.convert_to_tensor(
        [tf.complex(rho.full().real, rho.full().imag) for rho in rhos]
    )
    return tf_dms


def husimi_ops(hilbert_size, betas):
    """
    Constructs a list of TensorFlow operators for the Husimi Q function
    measurement at beta values.
    
    Args:
        hilbert_size (int): The hilbert size dimension for the operators
        betas (list/array): N complex values to construct the operator
        
    Returns:
        ops (:class:`tensorflow.Tensor`): A 3D tensor (N, hilbert_size, hilbert_size) of N
                                         operators
    """
    basis = []
    for beta in betas:
        op = qutip_coherent_dm(2*hilbert_size, beta)
        op = Qobj(op[:hilbert_size, :hilbert_size])
        basis.append(op)

    return dm_to_tf(basis)


def wigner_ops(hilbert_size, betas):
    """
    Constructs a list of TensorFlow operators for the Wigner function
    measurement at beta values.
    
    Args:
        hilbert_size (int): The hilbert size dimension for the operators
        betas (list/array): N complex values to construct the operator
        
    Returns:
        ops (:class:`tensorflow.Tensor`): A 3D tensor (N, hilbert_size, hilbert_size) of N
                                         operators
    """
    parity_op = sum([((-1)**i)*qutip_fock_dm(hilbert_size*2, i) for i in range(hilbert_size)])
    basis = []
    for beta in betas:
        D = displace(hilbert_size*2, beta)
        A = D*parity_op*D.dag()
        op = (A)*(2/np.pi)
        op = Qobj(op[:hilbert_size, :hilbert_size])
        basis.append(op)

    # ops = tf.convert_to_tensor([np.real(basis).astype(np.float64),
    #                            np.imag(basis).astype(np.float64)])
    return dm_to_tf(basis)




def tf_to_dm(rhos):
    """
    Convert a tensorflow density matrix to qutip density matrix

    Args:
        rhos (`tf.Tensor`): a 4D tensor (N, hilbert_size, hilbert_size)
                            representing density matrices        
        
    Returns:
        rho_gen (list of :class:`qutip.Qobj`): A list of N density matrices

    """
    rho_gen = [Qobj(rho.numpy()) for rho in rhos]
    return rho_gen


def clean_cholesky(img):
    """
    Cleans an input matrix to make it the Cholesky decomposition matrix T
    
    Args:
        img (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size, 2)
                           representing batch_size random outputs from a neural netowrk.
                           The last dimension is for separating the real and imaginary part

    Returns:
        T (`tf.Tensor`): a 3D tensor (N, hilbert_size, hilbert_size)
                           representing N T matrices
    """
    real = img[:, :, :, 0]
    imag = img[:, :, :, 1]

    diag_all = tf.linalg.diag_part(imag, k=0, padding_value=0)
    diags = tf.linalg.diag(diag_all)

    imag = imag - diags
    imag = tf.linalg.band_part(imag, -1, 0)
    real = tf.linalg.band_part(real, -1, 0)
    T = tf.complex(real, imag)
    return T


def density_matrix_from_T(tmatrix):
    """
    Gets density matrices from T matrices and normalizes them.
    
    Args:
        tmatrix (`tf.Tensor`): 3D tensor (N, hilbert_size, hilbert_size)
                           representing N valid T matrices

    Returns:
        rho (`tf.Tensor`): 3D tensor (N, hilbert_size, hilbert_size)
                           representing N density matrices
    """
    T = tmatrix
    T_dagger = tf.transpose(T, perm=[0, 2, 1], conjugate=True)
    proper_dm = tf.matmul(T, T_dagger)
    all_traces = tf.linalg.trace(proper_dm)
    all_traces = tf.reshape(1 / all_traces, (-1, 1))
    rho = tf.einsum("bij,bk->bij", proper_dm, all_traces)

    return rho


def convert_to_real_ops(ops):
    """
    Converts a batch of TensorFlow operators to something that a neural network
    can take as input.
    
    Args:
        ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size, hilbert_size) of N
                           measurement operators

    Returns:
        tf_ops (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size, 2*N) of N
                           measurement operators converted into real matrices
    """
    tf_ops = tf.transpose(ops, perm=[0, 2, 3, 1])
    tf_ops = tf.concat([tf.math.real(tf_ops), tf.math.imag(tf_ops)], axis=-1)
    return tf_ops


def convert_to_complex_ops(ops):
    """
    Converts a batch of TensorFlow operators to something that a neural network
    can take as input.
    
    Args:
        ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size, hilbert_size) of N
                           measurement operators

    Returns:
        tf_ops (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size, 2*N) of N
                           measurement operators converted into real matrices
    """
    shape = ops.shape
    num_points = shape[-1]
    tf_ops = tf.complex(ops[..., :int(num_points/2)], ops[..., int(num_points/2):])
    tf_ops = tf.transpose(tf_ops, perm=[0, 3, 1, 2])
    return tf_ops


def batched_expect(ops, rhos):
    """
    Calculates expectation values for a batch of density matrices
    for a batch of N sets of operators
    
    Args:
        ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size, hilbert_size) of N
                                         measurement operators
        rhos (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size)

    Returns:
        expectations (:class:`tensorflow.Tensor`): A 4D tensor (batch_size, N)
                                                   giving expectation values for the
                                                   N grid of operators for
                                                   all the density matrices (batch_size).
    """
    products = tf.einsum("bnij, bjk->bnik", ops, rhos)
    traces = tf.linalg.trace(products)
    expectations = tf.math.real(traces)
    return expectations


def tf_fidelity(A, B):
    """
    Calculated the fidelity between batches of tensors A and B
    """
    sqrtmA = tf.matrix_square_root(A)
    temp = tf.matmul(sqrtmA, B)
    temp2 = tf.matmul(temp, sqrtmA)
    fidel = tf.linalg.trace(tf.linalg.sqrtm(temp2))**2
    return tf.math.real(fidel)


def cat(hilbert_size, alpha=None, S=None, mu=None):
    """
    Generates a cat state. For a detailed discussion on the definition
    see `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_
    
    Args:
    -----
        N (int): Hilbert size dimension.
        alpha (complex64): Complex number determining the amplitude.
        S (int): An integer >= 0 determining the number of coherent states used
                 to generate the cat superposition. S = {0, 1, 2, ...}.
                 corresponds to {2, 4, 6, ...} coherent state superpositions.
        mu (int): An integer 0/1 which generates the logical 0/1 encoding of 
                  a computational state for the cat state.


    Returns:
    -------
        cat (:class:`qutip.Qobj`): Cat state density matrix
    """
    if alpha == None:
        alpha = random_alpha(2, 3)

    if S == None:
        S = np.random.randint(0, 3)

    if mu is None:
        mu = np.random.randint(0, 2)

    kend = 2 * S + 1
    cstates = 0 * (coherent(hilbert_size, 0))

    for k in range(0, int((kend + 1) / 2)):
        sign = 1

        if k >= S:
            sign = (-1) ** int(mu > 0.5)

        prefactor = np.exp(1j * (np.pi / (S + 1)) * k)

        cstates += sign * coherent(hilbert_size, prefactor * alpha * (-((1j) ** mu)))
        cstates += sign * coherent(hilbert_size, -prefactor * alpha * (-((1j) ** mu)))

    rho = cstates * cstates.dag()
    return rho.unit(), mu


def fock_dm(hilbert_size, n=None):
    """
    Generates a random fock state.
    
    Parameters
    ----------
    n : int
        The fock number
    Returns
    -------
    fock_dm: `qutip.Qobj`
        The density matrix as a quantum object.
    """
    if n == None:
        n = np.random.randint(1, hilbert_size/2 + 1)
    return qutip_fock_dm(hilbert_size, n), -1


def thermal_dm(hilbert_size, mean_photon_number=None):
    """
    Generates a random thermal state.

    Parameters
    ----------
    mean_photon_number: int
        The mean photon number for the thermal state.

    Returns
    -------
    thermal_dm: `qutip.Qobj`
        The density matrix as a quantum object.
    """
    if mean_photon_number == None:
        mean_photon_number = np.random.uniform(hilbert_size/2)
    return qutip_thermal_dm(hilbert_size, mean_photon_number), -1


def coherent_dm(hilbert_size, alpha=None):
    """
    Generates a random coherent state.

    Parameters
    ----------
    alpha: np.complex
        The displacement parameter. D(alpha)

    Returns
    -------
    rand_coherent: `qutip.Qobj`
        The density matrix as a quantum object.
    """
    if alpha == None:
        alpha = random_alpha(1e-6, 3)
    return qutip_coherent_dm(hilbert_size, alpha), -1


def gkp(hilbert_size, delta=None, mu = None):
    """Generates a GKP state
    """
    gkp = 0*coherent(hilbert_size, 0)

    c = np.sqrt(np.pi/2)

    if mu is None:
        mu = np.random.randint(2)

    if delta is None:
        delta = np.random.uniform(0.2, .50)

    zrange = range(-20, 20)

    for n1 in zrange:
        for n2 in zrange:        
            a = c*(2*n1 + mu + 1j*n2)
            alpha = coherent(hilbert_size, a)
            gkp += np.exp(-delta**2*np.abs(a)**2)*np.exp(-1j*c**2 * 2*n1 * n2)*alpha

    rho = gkp*gkp.dag()
    return rho.unit(), mu


def binomial(hilbert_size, S=None, N=None, mu=None):
    """
    Binomial code
    """
    if S == None:
        S = np.random.randint(1, 10)
    
    if N == None:
        Nmax = int((hilbert_size)/(S+1)) - 1
        try:
            N = np.random.randint(2, Nmax)
        except:
            N = Nmax

    if mu is None:
        mu = np.random.randint(2)

    c = 1/sqrt(2**(N+1))

    psi = 0*fock(hilbert_size, 0)

    for m in range(N):
        psi += c*((-1)**(mu*m))*np.sqrt(binom(N+1, m))*fock(hilbert_size, (S+1)*m)

    rho = psi*psi.dag()
    return rho.unit(), mu


def get_random_num_prob():
    states17 = [[(np.sqrt(7 - np.sqrt(17)))/np.sqrt(6), 0, 0, (np.sqrt(np.sqrt(17) - 1)/np.sqrt(6)), 0],
                [0, (np.sqrt(9 - np.sqrt(17))/np.sqrt(6)), 0, 0, (np.sqrt(np.sqrt(17) - 3)/np.sqrt(6))]]


    statesM = [[0.5458351325482939, -3.7726009161224436e-9, 4.849511177634774e-8, \
    -0.7114411727633639, -7.48481181758003e-8, -1.3146003192319789e-8, \
    0.44172510726665587, 1.1545802803733896e-8, 1.0609402576342428e-8, \
    -0.028182506843720707, -6.0233214626778965e-9, -6.392041552216322e-9, \
    0.00037641909140801935, -6.9186916801058116e-9], \
    [2.48926815257019e-9, -0.7446851186077535, -8.040831059521339e-9, \
    6.01942995399906e-8, -0.5706020908811399, -3.151900508005823e-8, \
    -7.384935824733578e-10, -0.3460030551087218, -8.485651303145757e-9, \
    -1.2114327561832047e-8, 0.011798401879159238, -4.660460771433317e-9, \
    -5.090374160706911e-9, -0.00010758601713550998]]


    statesP = [[0., 0.7562859301326029, 0., 0., -0.5151947804474741, \
    -0.20807866860791188, 0.12704803323656158, 0.05101928893751686, \
    0.3171198939841734], [-0.5583217426728544, -0.0020589109231194413, \
    0., -0.7014041964402703, -0.05583041652626998, 0.0005664728465725445, \
    -0.2755044401850055, -0.3333309025086189, 0.0785824556163142]]

    statesP2 = [[-0.5046617350158988, 0.08380989527942606, -0.225295417417812, 0., \
    -0.45359477373452817, -0.5236866813756252, 0.2523308675079494, 0., \
    0.09562538828178244, 0.2172849136874009, 0., 0., 0., \
    -0.2793663175980869, -0.08280858231312467, -0.05106696128137072], \
    [-0.0014249418817930378, 0.5018692341095683, 0.4839749920101922, \
    -0.3874886488913531, 0.055390715144453026, -0.25780190053922486, \
    -0.08970154713375252, -0.1892386424818236, 0.10840637100094529, \
    -0.19963901508324772, -0.41852779130900664, -0.05747247660559087, 0., \
    -0.0007888071131354318, -0.1424131123943283, -0.0001441905475623907]]


    statesM2 = [[-0.45717455741713664, \
    np.complex(-1.0856965103853774e-6,1.3239037829080093e-6), \
    np.complex(-0.35772784377291084,-0.048007740168066144), \
    np.complex(-3.5459165445315755e-6,0.000012571453643232864), \
    np.complex(-0.5383420820794502,-0.24179040513272307), \
    np.complex(9.675641330014822e-7,4.569566899500361e-6), \
    np.complex(0.2587482691377581,0.313044506480362), \
    np.complex(4.1979351791851435e-6,-1.122460690803522e-6), \
    np.complex(-0.11094500303308243,0.20905585817734396), \
    np.complex(-1.1837814323046472e-6,3.8758497675466054e-7), \
    np.complex(0.1275629945870373,-0.1177987279989385), \
    np.complex(-2.690647673469878e-6,-3.6519804939862998e-6), \
    np.complex(0.12095531973074151,-0.19588735180644176), \
    np.complex(-2.6588791126371675e-6,-6.058292629669095e-7), \
    np.complex(0.052905370429015865,-0.0626791930782206), \
    np.complex(-1.6615538648519722e-7,6.756126951837809e-8), \
    np.complex(0.016378329200891946,-0.034743342821208854), \
    np.complex(4.408946495377283e-8,2.2826415255126898e-8), \
    np.complex(0.002765352838800482,-0.010624191776867055), \
    6.429253878486627e-8, \
    np.complex(0.00027095836439738105,-0.002684435917226972), \
    np.complex(1.1081202749445256e-8,-2.938812506852636e-8), \
    np.complex(-0.000055767533641099717,-0.000525444354381421), \
    np.complex(-1.0776974926155464e-8,-2.497769263148397e-8), \
    np.complex(-0.000024992489351114305,-0.00008178444317382933), \
    np.complex(-1.5079116121444066e-8,-2.0513760149701907e-8), \
    np.complex(-5.64035228941742e-6,-0.000010297667130821428), \
    np.complex(-1.488452012610573e-8,-1.7358623165948514e-8), \
    np.complex(-8.909884885392901e-7,-1.04267002748775e-6), \
    np.complex(-1.2056784102984098e-8,-1.2210951690230782e-8)], [0, \
    0.5871298855433338, \
    np.complex(-3.3729618710801137e-6,2.4152360811650373e-6), \
    np.complex(-0.5233926069798007,-0.13655786303346068), \
    np.complex(-4.623380373113224e-6,0.000010362902695259763), \
    np.complex(-0.17909656013941788,-0.11916639160269833), \
    np.complex(-3.399720873431807e-6,-7.125008373682292e-7), \
    np.complex(0.04072119358712736,-0.3719310475303641), \
    np.complex(-7.536125619789242e-6,1.885248226837573e-6), \
    np.complex(-0.11393851510585044,-0.3456924286310791), \
    np.complex(-2.3915763815197452e-6,-4.2406689395594674e-7), \
    np.complex(0.12820184730203607,0.0935942533049232), \
    np.complex(-1.5407293261691393e-6,-2.4673669087089514e-6), \
    np.complex(-0.012272903377715643,-0.13317144020065683), \
    np.complex(-1.1260776123106269e-6,-1.6865728072273087e-7), \
    np.complex(-0.01013345155253134,-0.0240812705564227), \
    np.complex(0.,-1.4163391111474348e-7), \
    np.complex(-0.003213070562510137,-0.012363639898516247), \
    np.complex(-1.0619280312362908e-8,-1.2021213613319027e-7), \
    np.complex(-0.002006756716685063,-0.0026636832583059812), \
    np.complex(0.,-4.509035934797572e-8), \
    np.complex(-0.00048585160444833446,-0.0005014735884977489), \
    np.complex(-1.2286988061034212e-8,-2.1199721851825594e-8), \
    np.complex(-0.00010897007463988193,-0.00007018240288615613), \
    np.complex(-1.2811279935244964e-8,-1.160553871672415e-8), \
    np.complex(-0.00001785800494916693,-6.603027186486886e-6), \
    -1.1639448324793031e-8, \
    np.complex(-2.4097385882316104e-6,-3.5223103057306496e-7), \
    -1.0792272866841885e-8, \
    np.complex(-2.597671478115077e-7,2.622928060603902e-8)]]

    all_num_codes = [states17, statesM, statesM2, statesP, statesP2]

    probs = all_num_codes[np.random.randint(len(all_num_codes))]
    return probs


def num(hilbert_size, probs=None, mu=None, alpha_range=3):
    """
    number code
    """
    if mu is None:
        mu = np.random.randint(2)

    state = fock(hilbert_size, 0)*0

    if probs is None:
        probs = get_random_num_prob()

    for n, p in enumerate(probs[mu]):
        state += p*fock(hilbert_size, n)    
    rho = state*state.dag()
    return rho.unit(), mu

def random(hilbert_size, density=None):
    """
    Generates a completely random but physical state
    
    Parameters
    ----------
    density : int
        The density of the state
    Returns
    -------
    rand_dm: `qutip.Qobj`
        The density matrix as a quantum object.
    """
    if density == None:
        density = np.random.uniform(0.05, 1)
    return rand_dm(hilbert_size, density), -1



def add_state_noise(dm, sigma=0.01, sparsity=0.01):
    """
    Adds a random density matrices to the input state.
    
    .. math::
        \rho_{mixed} = \sigma \rho_0 + (1 - \sigma)\rho_{rand}$

    Args:
    ----
        dm (`qutip.Qobj`): Density matrix of the input pure state
        sigma (float): the mixing parameter specifying the pure state probability
        sparsity (float): the sparsity of the random density matrix
    
    Returns:
    -------
        rho (`qutip.Qobj`): the mixed state density matrix
    """
    hilbertsize = dm.shape[0]
    rho  = (1 - sigma)*dm + sigma*(rand_dm(hilbertsize, sparsity))
    rho = rho/rho.tr()
    return rho

def measure(alpha, rho=None):
    """
    Measures the photon number statistics for state rho when displaced
    by angle alpha.
    
    Parameters
    ----------    
    alpha: np.complex
        A complex displacement.
    Returns
    -------
    population: ndarray
        A 1D array for the probabilities for populations.
    """
    hilbertsize = rho.shape[0]
    D = displace(hilbertsize, -alpha)
    rho_disp = D*rho*D.dag()
    populations = np.real(np.diagonal(rho_disp.full()))
    return np.array(populations).reshape(-1)


def generalized_q(rho, xvec, yvec):
    hilbertsize = rho.shape[0]
    q = np.zeros(shape = (len(xvec), len(yvec), hilbertsize))
    for i, p in enumerate(yvec):
        for j, x in enumerate(xvec):
            beta = (x + 1j*p)/np.sqrt(2)
            q[i, j] = measure(beta, rho)
    return q



def visualize(original, augmented, title=""):
    """
    Visualize augumented images
    """
    fig, ax = plt.subplots(1, 2,)
    ax[0].set_title('Original image')

    ax[0].imshow(original[..., 0], cmap="RdBu",
                 vmin=np.min(original[..., 0]),
                 vmax = np.max(original[..., 0]))

    ax[1].set_title('Augmented image')
    ax[1].imshow(augmented[..., 0],
                 vmin=np.min(augmented[..., 0]),
                 vmax = np.max(augmented[..., 0]),
                 cmap="RdBu")
    plt.suptitle(title)
    plt.show()



