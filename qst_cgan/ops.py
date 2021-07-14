"""
Utility function and operations
"""
import numpy as np


from qutip import Qobj

import tensorflow as tf


def batched_expect(ops, rhos):
    """
    Calculates expectation values for a batch of density matrices
    for a set of operators.

    Args:
        ops (`tf.Tensor`): a tensor of shape (batch_size, N, hilbert_size,
                                                             hilbert_size) 
                           of N measurement operators.
        rhos (`tf.Tensor`): a tensor (batch_size, hilbert_size, hilbert_size).

    Returns:
        expectations (:class:`tf.Tensor`): A tensor shaped as (batch_size, N)
                                           representing expectation values for
                                           the N operators for all the density
                                           matrices (batch_size).
    """
    products = tf.einsum("bnij, bjk->bnik", ops, rhos)
    traces = tf.linalg.trace(products)
    expectations = tf.math.real(traces)
    return expectations


def random_alpha(radius, inner_radius=0):
    """
    Generates random complex numbers within a circle.

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
        tf_dms (:class:`tf.Tensor`): A tensor of shape (N, hilbert_size,
                                                           hilbert_size)
                                     of N density matrices.
    """
    tf_dms = tf.convert_to_tensor(
        [tf.complex(rho.full().real, rho.full().imag) for rho in rhos]
    )
    return tf_dms


def tf_to_dm(rhos):
    """
    Convert a tensorflow density matrix to qutip density matrix

    Args:
        rhos (`tf.Tensor`): a tensor of shape (N, hilbert_size, hilbert_size)
                            representing N density matrices.

    Returns:
        rho_gen (list of :class:`qutip.Qobj`): A list of N density matrices.

    """
    rho_gen = [Qobj(rho.numpy()) for rho in rhos]
    return rho_gen


def clean_cholesky(img):
    """
    Cleans an input matrix to make it the Cholesky decomposition matrix T

    Args:
        img (`tf.Tensor`): a tensor of shape (batch_size, hilbert_size,
                                                          hilbert_size, 2)
                           representing random outputs from a neural netowrk.
                           The last dimension is for separating the real and
                           imaginary part.

    Returns:
        T (`tf.Tensor`): a 3D tensor (N, hilbert_size, hilbert_size)
                           representing N matrices used for Cholesky decomp.
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
        tmatrix (`tf.Tensor`): A tensor (N, hilbert_size, hilbert_size)
                               representing N valid T matrices.

    Returns:
        rho (`tf.Tensor`): A tensor of shape (N, hilbert_size, hilbert_size)
                           representing N density matrices.
    """
    T = tmatrix
    T_dagger = tf.transpose(T, perm=[0, 2, 1], conjugate=True)
    proper_dm = tf.matmul(T_dagger, T)
    all_traces = tf.linalg.trace(proper_dm)
    all_traces = tf.reshape(1 / all_traces, (-1, 1))
    rho = tf.einsum("bij,bk->bij", proper_dm, all_traces)

    return rho


def convert_to_real_ops(ops):
    """
    Converts a batch of TensorFlow operators to something that a neural network
    can take as input.

    Args:
        ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size,
                                                       hilbert_size)
                           of N measurement operators.

    Returns:
        tf_ops (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size,
                                                       hilbert_size, 2*N)
                              of N measurement operators converted into real
                              matrices.
    """
    tf_ops = tf.transpose(ops, perm=[0, 2, 3, 1])
    tf_ops = tf.concat([tf.math.real(tf_ops), tf.math.imag(tf_ops)], axis=-1)
    return tf_ops


def convert_to_complex_ops(ops):
    """
    Converts a batch of TensorFlow operators to something that a neural network
    can take as input.

    Args:
        ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size,
                                                       hilbert_size)
                           of N measurement operators.

    Returns:
        tf_ops (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size,
                                                       hilbert_size, 2*N)
                              of N measurement operators converted into real
                              matrices.
    """
    shape = ops.shape
    num_points = shape[-1]
    tf_ops = tf.complex(
        ops[..., : int(num_points / 2)], ops[..., int(num_points / 2) :]
    )
    tf_ops = tf.transpose(tf_ops, perm=[0, 3, 1, 2])
    return tf_ops


def tf_fidelity(A, B):
    """Calculates the fidelity between tensors A and B.

    Args:
        A, B (tf.Tensor): List of tensors (hilbert_size, hilbert_size).

    Returns:
        float: Fidelity between A and B
    """
    sqrtmA = tf.matrix_square_root(A)
    temp = tf.matmul(sqrtmA, B)
    temp2 = tf.matmul(temp, sqrtmA)
    fidel = tf.linalg.trace(tf.linalg.sqrtm(temp2)) ** 2
    return tf.math.real(fidel)
