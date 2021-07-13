"""
Bigger GAN network
"""
import numpy as np


import tensorflow as tf
from tensorflow.keras.layers import Input
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


from qst_cgan.ops import (
    random_alpha,
    dm_to_tf,
    husimi_ops,
    tf_to_dm,
    clean_cholesky,
    density_matrix_from_T,
    batched_expect,
    convert_to_real_ops,
    convert_to_complex_ops,
)


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
        return prefactor * batched_expect(ops, rhos)


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
    initializer = tf.random_normal_initializer(0.0, 0.02)

    n = int(hilbert_size / 2)

    ops = tf.keras.layers.Input(
        shape=[hilbert_size, hilbert_size, num_points * 2], name="operators"
    )
    inputs = tf.keras.Input(shape=(num_points), name="inputs")

    x = tf.keras.layers.Dense(
        16 * 16 * 2,
        use_bias=False,
        kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
    )(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Reshape((16, 16, 2))(x)

    x = tf.keras.layers.Conv2DTranspose(
        64, 4, use_bias=False, strides=2, padding="same", kernel_initializer=initializer
    )(x)
    x = tfa.layers.InstanceNormalization(axis=3)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(
        64, 4, use_bias=False, strides=1, padding="same", kernel_initializer=initializer
    )(x)
    x = tfa.layers.InstanceNormalization(axis=3)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(
        32, 4, use_bias=False, strides=1, padding="same", kernel_initializer=initializer
    )(x)

    x = tf.keras.layers.Conv2DTranspose(
        2, 4, use_bias=False, strides=1, padding="same", kernel_initializer=initializer
    )(x)
    x = DensityMatrix()(x)
    complex_ops = convert_to_complex_ops(ops)
    # prefactor = (0.25*g**2/np.pi)
    prefactor = 1.0
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
    initializer = tf.random_normal_initializer(0.0, 0.002)

    inp = tf.keras.layers.Input(shape=[num_points], name="input_image")
    tar = tf.keras.layers.Input(shape=[num_points], name="target_image")
    ops = tf.keras.layers.Input(
        shape=[hilbert_size, hilbert_size, num_points * 2], name="operators"
    )
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
    x = tf.keras.layers.Dense(128, kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Dense(128, kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(64, kernel_initializer=initializer, activation="relu")(x)
    x = tf.keras.layers.Dense(64)(x)

    return tf.keras.Model(inputs=[ops, inp, tar], outputs=x)


def discriminator_loss(disc_real_output, disc_generated_output):
    """Discriminator loss function

    Args:
        disc_real_output (`tf.Tensor`): Output of the discriminator when it was shown the actual data
        as the target
        disc_generated_output (`tf.Tensor`): Output of the discriminator when it was shown the generated data
        as the target
    Returns:
        total_disc_loss (tf.float): The loss value for the discriminator which is optimized

    """
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target, LAMBDA=0.0):
    """Summary

    Args:
        disc_generated_output (`tf.Tensor`): Output of the discriminator when it was shown the generated data
        as the target
        gen_output (`tf.Tensor`):
        target (TYPE): Description
        LAMBDA (float, optional): Description

    Returns:
        TYPE: Description
    """
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + LAMBDA * l1_loss

    return total_gen_loss, gan_loss, l1_loss


def train_step(A, x):
    """Takes one step of training for the full A matrix and x statistics

    Args:
        A (TYPE): Description
        x (TYPE): Description
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator([A, x], training=True)

        disc_real_output = discriminator([A, x, x], training=True)
        disc_generated_output = discriminator([A, x, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, x
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )


def train_gan(
    A,
    x,
    rho_true=None,
    tol=None,
    generator=None,
    discriminator=None,
    max_iterations=100,
    log_interval=20,
    lam=0.0,
    lr=None,
):
    tf.keras.backend.clear_session()
    hilbert_size = A.shape[1]
    num_points = int(A.shape[-1] / 2)

    if tol is None:
        tol = 1e-4

    if generator is None:
        generator = Generator(hilbert_size, num_points)

    if discriminator is None:
        discriminator = Discriminator(hilbert_size, num_points)

    density_layer_idx = None

    for i, layer in enumerate(generator.layers):
        if "density_matrix" in layer._name:
            density_layer_idx = i
            break
    model_dm = tf.keras.Model(
        inputs=generator.input, outputs=generator.layers[density_layer_idx].output
    )

    if lr == None:
        initial_learning_rate = 0.002
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.96, staircase=False
        )

    else:
        lr_schedule = lr

    generator_optimizer = tf.keras.optimizers.Adam(lr_schedule, 0.5, 0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr_schedule, 0.5, 0.5)

    def train_step(A, x):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator([A, x], training=True)

            disc_real_output = discriminator([A, x, x], training=True)
            disc_generated_output = discriminator([A, x, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, x, LAMBDA=lam
            )
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

    f_list = []
    gen_dm = model_dm([A, x], training=False)
    rho_gen = tf_to_dm(gen_dm)

    states = [rho_gen[0]]
    pbar = tqdm(range(max_iterations))
    skip = False

    for i in pbar:
        if skip:
            pbar.close()
            break

        else:
            train_step(A, x)
            gen_dm = model_dm([A, x], training=False)
            rho_gen = tf_to_dm(gen_dm)[0]
            states.append(rho_gen)
            if rho_true is not None:
                if i % log_interval == 0:
                    f_mean_10 = np.mean([fidelity(rho_true, s) for s in states[-10:]])
                    f_mean_20 = np.mean(
                        [fidelity(rho_true, s) for s in states[-20:-10]]
                    )
                    if np.abs(f_mean_10 - f_mean_20) < tol:
                        skip = True
                f = fidelity(rho_true, states[-1])
                f_list.append(f)
                pbar.set_description("Fidelity QST-CGAN {}".format(f))
            else:
                pass
                pbar.set_description("Pass rho_true to show fidelity to a target state")

    return f_list, model_dm, states
