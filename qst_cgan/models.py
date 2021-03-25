import tensorflow as tf
from tensorflow.keras.layers import Input
import tensorflow_addons as tfa

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_addons as tfa

from qutip.visualization import hinton, plot_wigner
from qutip.wigner import qfunc, wigner
from qutip import Qobj, fidelity, qeye, destroy
from qutip.states import coherent_dm, thermal_dm, coherent, fock_dm
from qutip.random_objects import rand_dm
from qutip import expect
from qutip import coherent_dm, coherent, expect


from tqdm.auto import tqdm
import numpy as np
import h5py

from qst_cgan.ops import expect as tf_expect
from qst_cgan.ops import (random_alpha, dm_to_tf, husimi_ops, tf_to_dm, clean_cholesky, density_matrix_from_T,
                 batched_expect, convert_to_real_ops, convert_to_complex_ops)





def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer,
                             kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                             use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def Classifier():
    inp = tf.keras.layers.Input(shape=[32, 32, 1], name='input_image')

    x = tf.keras.layers.Conv2D(32, 3, strides=1,
                               use_bias=False,
                              )(inp)
    x = tf.keras.layers.LeakyReLU()(x)
    

    x = tf.keras.layers.Conv2D(32, 3, strides=1,
                               use_bias=False,
                              )(x)
    x = tf.keras.layers.LeakyReLU()(x)


    x = tf.keras.layers.Conv2D(32, 3, strides=2,
                               use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    

    x = tf.keras.layers.Conv2D(64, 3, strides=1,
                              use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.GaussianNoise(0.002)(x)
    x = tf.keras.layers.Dropout(0.4)(x)    
    
    x = tf.keras.layers.Conv2D(64, 3, strides=1,
                              use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(64, 3, strides=2,
                              use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Dropout(0.4)(x)    
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Dropout(0.4)(x)    
    
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Dense(7)(x)

    return tf.keras.Model(inputs=inp, outputs=x)



class DensityMatrix(tf.keras.layers.Layer):
    """
    Density matrix layer that cleans the input matrix into a Cholesky matrix
    and then constructs the density matrix for the state
    """
    def __init__(self):
        super(DensityMatrix, self).__init__()

    def call(self, inputs, training=False):
        """
        The call function which applies the Cholesky decomposition

        Args:
            inputs (`tf.Tensor`): a 4D real valued tensor (batch_size, hilbert_size, hilbert_size, 2)
                           representing batch_size random outputs from a neural netowrk.
                           The last dimension is for separating the real and imaginary part

        Returns:
            dm (`tf.Tensor`): A 3D complex valued tensor (batch_size, hilbert_size, hilbert_size)
                              representing valid density matrices from a Cholesky decomposition of the
                              cleaned input
        """
        T = clean_cholesky(inputs)
        return density_matrix_from_T(T)




class Expectation(tf.keras.layers.Layer):
    """
    Expectation layer that calculates expectation values for a set of operators on a batch of rhos.
    You can specify different sets of operators for each density matrix in the batch.
    """
    def __init__(self):
        super(Expectation, self).__init__()

    def call(self, ops, rhos, prefactor=1):
        """Expectation function call
        
        Args:
            ops (`tf.Tensor`): a 4D complex tensor (batch_size, N, hilbert_size, hilbert_size) of N
                                         measurement operators
            rhos (`tf.Tensor`): a 4D complex tensor (batch_size, hilbert_size, hilbert_size)

        Returns:
            expectations (:class:`tf.Tensor`): A 2D tensor (batch_size, N)
                                                   giving expectation values for the
                                                   N grid of operators for
                                                   all the density matrices (batch_size).
        """
        return prefactor*batched_expect(ops, rhos)



def GeneratorSimple(hilbert_size, num_points, ops_batch, noise=0.):
    """
    A tensorflow generative model which can be called as 
                  >> generator([A, x])
    where A is the set of all measurement operators
    transformed into the shape (batch_size, hilbert_size, hilbert_size, num_points*2)
    This can be done using the function `convert_to_real_ops` which
    takes a set of complex operators shaped as (batch_size, num_points, hilbert_size, hilbert_size)
    and converts it to this format which is easier to run convolution operations on.

    x is the measurement statistics (frequencies) represented by a vector of shape
    [batch_size, num_points] where we consider num_points different operators and their
    expectation values.

    Args:
        hilbert_size (int): Hilbert size of the output matrix
                            This needs to be 32 now. We can adjust 
                            the network architecture to allow it to
                            automatically change its outputs according
                            to the hilbert size in future
        num_points (int): Number of different measurement operators
    
    Returns:
        generator: A TensorFlow model callable as
        >> generator([A, x])
    
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    n = int(hilbert_size/2)
    
    inputs = tf.keras.Input(shape=(num_points), name = "inputs")
    

    x = tf.keras.layers.Dense(16*16*2, use_bias=False,
                              kernel_initializer = tf.random_normal_initializer(0., 0.02),
                             )(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Reshape((16, 16, 2))(x)

    x = tf.keras.layers.Conv2DTranspose(64, 4, use_bias=False,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=3)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 4, use_bias=False,
                                        strides=1,
                                        padding='same',
                                        kernel_initializer=initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=3)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(32, 4, use_bias=False,
                                        strides=1,
                                        padding='same',
                                        kernel_initializer=initializer)(x)
    # x = tfa.layers.InstanceNormalization(axis=3)(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    # y = tf.keras.layers.Conv2D(8, 5, padding='same')(ops)
    # out = x
    # x = tf.keras.layers.concatenate([x, y])
    x = tf.keras.layers.Conv2DTranspose(2, 4, use_bias=False,
                                    strides=1,
                                    padding='same',
                                    kernel_initializer=initializer)(x)
    x = DensityMatrix()(x)
    
    x = batched_expect(ops_batch, x)

    x = tf.keras.layers.GaussianNoise(noise)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Generator(hilbert_size, num_points, noise=None):
    """
    A tensorflow generative model which can be called as 
                  >> generator([A, x])
    where A is the set of all measurement operators
    transformed into the shape (batch_size, hilbert_size, hilbert_size, num_points*2)
    This can be done using the function `convert_to_real_ops` which
    takes a set of complex operators shaped as (batch_size, num_points, hilbert_size, hilbert_size)
    and converts it to this format which is easier to run convolution operations on.

    x is the measurement statistics (frequencies) represented by a vector of shape
    [batch_size, num_points] where we consider num_points different operators and their
    expectation values.

    Args:
        hilbert_size (int): Hilbert size of the output matrix
                            This needs to be 32 now. We can adjust 
                            the network architecture to allow it to
                            automatically change its outputs according
                            to the hilbert size in future
        num_points (int): Number of different measurement operators
    
    Returns:
        generator: A TensorFlow model callable as
        >> generator([A, x])
    
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    n = int(hilbert_size/2)
    
    ops = tf.keras.layers.Input(shape=[hilbert_size, hilbert_size, num_points*2],
                                name='operators')
    inputs = tf.keras.Input(shape=(num_points), name = "inputs")
    

    x = tf.keras.layers.Dense(16*16*2, use_bias=False,
                              kernel_initializer = tf.random_normal_initializer(0., 0.02),
                             )(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Reshape((16, 16, 2))(x)

    x = tf.keras.layers.Conv2DTranspose(64, 4, use_bias=False,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=3)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 4, use_bias=False,
                                        strides=1,
                                        padding='same',
                                        kernel_initializer=initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=3)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(32, 4, use_bias=False,
                                        strides=1,
                                        padding='same',
                                        kernel_initializer=initializer)(x)
    # x = tfa.layers.InstanceNormalization(axis=3)(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    # y = tf.keras.layers.Conv2D(8, 5, padding='same')(ops)
    # out = x
    # x = tf.keras.layers.concatenate([x, y])
    x = tf.keras.layers.Conv2DTranspose(2, 4, use_bias=False,
                                    strides=1,
                                    padding='same',
                                    kernel_initializer=initializer)(x)
    x = DensityMatrix()(x)
    complex_ops = convert_to_complex_ops(ops)
    # prefactor = (0.25*g**2/np.pi)
    prefactor = 1.
    x = Expectation()(complex_ops, x, prefactor)
    x = tf.keras.layers.GaussianNoise(noise)(x)

    return tf.keras.Model(inputs=[ops, inputs], outputs=x)


def Discriminator(hilbert_size, num_points):
    """A tensorflow generative model which can be called as 
                  >> discriminator([A, x, y])
    where A is the set of all measurement operators
    transformed into the shape (batch_size, hilbert_size, hilbert_size, num_points*2)
    This can be done using the function `convert_to_real_ops` which
    takes a set of complex operators shaped as (batch_size, num_points, hilbert_size, hilbert_size)
    and converts it to this format which is easier to run convolution operations on.

    x is the measurement statistics (frequencies) represented by a vector of shape
    [batch_size, num_points] where we consider num_points different operators and their
    expectation values.

    y can be the generated statistics by the generator or x. The discriminators
    job is to give a high probability of match for the true data as the target (x)
    and low probability for the generated data (y)

    Args:
        hilbert_size (int): Hilbert size of the output matrix
                            This needs to be 32 now. We can adjust 
                            the network architecture to allow it to
                            automatically change its outputs according
                            to the hilbert size in future
        num_points (int): Number of different measurement operators
     
    Returns:
        discriminator: discriminator which can be called as 
                        >> discriminator([A, x, y])
    """
    initializer = tf.random_normal_initializer(0., 0.002)

    inp = tf.keras.layers.Input(shape=[num_points],
                              name='input_image')
    tar = tf.keras.layers.Input(shape=[num_points],
                              name='target_image')
    ops = tf.keras.layers.Input(shape=[hilbert_size, hilbert_size, num_points*2],
                                name='operators')

    # y = tf.keras.layers.Conv2D(1, 4, 2, activation="relu", padding='same')(ops)
    # y = tf.keras.layers.Flatten()(y)

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
    x = tf.keras.layers.Dense(128, kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    # x = tf.keras.layers.concatenate([x, y]) # (bs, 256, 256, channels*2)
    # x = tf.keras.layers.Dense(128, activation="relu",
    #                           kernel_initializer=initializer)(x)
    x = tf.keras.layers.Dense(128, kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(64, kernel_initializer=initializer,
                             activation="relu")(x)
    x = tf.keras.layers.Dense(64)(x)


    return tf.keras.Model(inputs=[ops, inp, tar], outputs=x)









def GeneratorCNN(hilbert_size, dim, noise=None):
    """
    A tensorflow generative model which can be called as 
                  >> generator([A, x])
    where A is the set of all measurement operators
    transformed into the shape (batch_size, hilbert_size, hilbert_size, num_points*2)
    This can be done using the function `convert_to_real_ops` which
    takes a set of complex operators shaped as (batch_size, num_points, hilbert_size, hilbert_size)
    and converts it to this format which is easier to run convolution operations on.

    x is the measurement statistics (frequencies) represented by a vector of shape
    [batch_size, xdim, ydim] where we consider (xdim, ydim) shaped 
    phase-space images of expectation values.

    Args:
        hilbert_size (int): Hilbert size of the output matrix
                            This needs to be 32 now. We can adjust 
                            the network architecture to allow it to
                            automatically change its outputs according
                            to the hilbert size in future
        dim (int): Dimension of the grid for measurement
    
    Returns:
        generator: A TensorFlow model callable as
        >> generator([A, x])
    
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    n = int(hilbert_size/2)
    num_points = dim*dim
    ops = tf.keras.layers.Input(shape=[hilbert_size, hilbert_size, num_points*2],
                                name='operators')
    inputs = tf.keras.Input(shape=(dim, dim, 1), name = "inputs")
    
    #x = tf.keras.layers.Conv2D(64, 3, activation="relu")(inputs)
    #x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3)(inputs)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(64, 3)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(64, 5, 2)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(64, 5, 2)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16*16*2, use_bias=False,
                              kernel_initializer = tf.random_normal_initializer(0., 0.02),
                             )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Reshape((16, 16, 2))(x)

    x = tf.keras.layers.Conv2DTranspose(16, 4, use_bias=False,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer)(x)
    x = tfa.layers.InstanceNormalization(axis=3)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    # x = tf.keras.layers.Conv2DTranspose(64, 4, use_bias=False,
    #                                     strides=1,
    #                                     padding='same',
    #                                     kernel_initializer=initializer)(x)
    # x = tfa.layers.InstanceNormalization(axis=3)(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    # x = tf.keras.layers.Conv2DTranspose(32, 4, use_bias=False,
    #                                     strides=1,
    #                                     padding='same',
    #                                     kernel_initializer=initializer)(x)

    x = tf.keras.layers.Conv2DTranspose(2, 4, use_bias=False,
                                    strides=1,
                                    padding='same',
                                    kernel_initializer=initializer)(x)
    x = DensityMatrix()(x)

    complex_ops = convert_to_complex_ops(ops)
    # prefactor = (0.25*g**2/np.pi)
    prefactor = 1.
    x = Expectation()(complex_ops, x, prefactor)
    x = tf.keras.layers.Reshape((dim, dim, 1))(x)
    x = tf.keras.layers.GaussianNoise(noise)(x)

    return tf.keras.Model(inputs=[ops, inputs], outputs=x)




def DiscriminatorCNN(hilbert_size, dim):
    """A tensorflow generative model which can be called as 
                  >> discriminator([A, x, y])
    where A is the set of all measurement operators
    transformed into the shape (batch_size, hilbert_size, hilbert_size, num_points*2)
    This can be done using the function `convert_to_real_ops` which
    takes a set of complex operators shaped as (batch_size, num_points, hilbert_size, hilbert_size)
    and converts it to this format which is easier to run convolution operations on.

    x is the measurement statistics (frequencies) represented by a vector of shape
    [batch_size, num_points] where we consider num_points different operators and their
    expectation values.

    y can be the generated statistics by the generator or x. The discriminators
    job is to give a high probability of match for the true data as the target (x)
    and low probability for the generated data (y)

    Args:
        hilbert_size (int): Hilbert size of the output matrix
                            This needs to be 32 now. We can adjust 
                            the network architecture to allow it to
                            automatically change its outputs according
                            to the hilbert size in future
        num_points (int): Number of different measurement operators
     
    Returns:
        discriminator: discriminator which can be called as 
                        >> discriminator([A, x, y])
    """
    initializer = tf.random_normal_initializer(0., 0.002)

    inp = tf.keras.layers.Input(shape=[dim, dim, 1],
                              name='input_image')
    tar = tf.keras.layers.Input(shape=[dim, dim, 1],
                              name='target_image')

    num_points = dim*dim
    ops = tf.keras.layers.Input(shape=[hilbert_size, hilbert_size, num_points*2],
                                name='operators')

    # y = tf.keras.layers.Conv2D(1, 4, 2, activation="relu", padding='same')(ops)
    # y = tf.keras.layers.Flatten()(y)

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)


    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(128, kernel_initializer=initializer)(x)
    # x = tf.keras.layers.LeakyReLU()(x)

    # x = tf.keras.layers.Dense(128, kernel_initializer=initializer)(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    # x = tf.keras.layers.Dense(64, kernel_initializer=initializer,
    #                          activation="relu")(x)
    # x = tf.keras.layers.Dense(dim*dim)(x)
    # x = tf.keras.layers.Reshape((dim, dim, 1))(x)

    return tf.keras.Model(inputs=[ops, inp, tar], outputs=x)

