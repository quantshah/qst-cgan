"""
Bigger GAN network
"""

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

from IPython.display import clear_output

from tqdm.auto import tqdm
import numpy as np
import h5py

from qst_cgan.ops import expect as tf_expect
from qst_cgan.ops import (random_alpha, dm_to_tf, husimi_ops, tf_to_dm, clean_cholesky, density_matrix_from_T,
                 batched_expect, convert_to_real_ops, convert_to_complex_ops)

from qst_cgan.models import Generator, DensityMatrix, Expectation, Discriminator, GeneratorSimple


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

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target,
    lam_l1 = 0., lam_gan=1):
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

    total_gen_loss = lam_gan*gan_loss + lam_l1 * l1_loss

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

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, x)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))


def train_GAN(A, x, rho,
              generator=None, discriminator=None,
              max_iterations=100,
              lam=0., gan_lam=1, lr=None, noise=0,
              fidelity_list=None, states_list = None, optimizer=None,
              log_interval=100,
              patience=2, tol=1e-3):
    tf.keras.backend.clear_session()
    hilbert_size = A.shape[1]
    num_points = int(A.shape[-1]/2)

    if tol is None:
        tol = 0
            
    if generator is None:
        generator = Generator(hilbert_size, num_points, noise=noise)

    if discriminator is None:
        discriminator = Discriminator(hilbert_size, num_points)
    
    density_layer_idx = None

    for i, layer in enumerate(generator.layers): 
        if "density_matrix" in layer._name:
            density_layer_idx = i
            break
    model_dm = tf.keras.Model(inputs=generator.input,
                              outputs=generator.layers[density_layer_idx].output)

    if lr == None:
        initial_learning_rate = 0.0002
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                             decay_steps=10000,
                                                             decay_rate=0.96,
                                                             staircase=True)
    else:
        lr_schedule = lr

    if optimizer == None:
        generator_optimizer = tf.keras.optimizers.Adam(lr_schedule, 0.5, 0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(lr_schedule, 0.5, 0.5)
    else:
        generator_optimizer = optimizer
        discriminator_optimizer = optimizer

    def train_step(A, x):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator([A, x], training=True)

            disc_real_output = discriminator([A, x, x], training=True)
            disc_generated_output = discriminator([A, x, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output,
                gen_output, x,
                lam_l1=lam, lam_gan=gan_lam)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   discriminator.trainable_variables)

        slopes = tf.convert_to_tensor([tf.sqrt(tf.reduce_sum(s**2)) for s in discriminator_gradients])
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator([A, x], training=True)

            disc_real_output = discriminator([A, x, x], training=True)
            disc_generated_output = discriminator([A, x, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, x,
                lam_l1=lam, lam_gan=gan_lam)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output) + 10*gradient_penalty

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   discriminator.trainable_variables)


        generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  discriminator.trainable_variables))

        return gen_total_loss, disc_loss
        

    gen_dm = model_dm([A, x], training=False)
    rho_gen = tf_to_dm(gen_dm)


    if states_list == None:
        states_list = [rho_gen[0]]
    else:
        states_list.append(rho_gen[0])

    if fidelity_list == None:
        fidelity_list = []
        fidelity_list.append(fidelity(rho, states_list[-1]))
    else:
        fidelity_list.append(fidelity(rho, states_list[-1]))

    pbar = tqdm(range(max_iterations))


    skip_count = 0
    current_mean = 0

    for i in pbar:
        train_step(A, x)
        gen_dm = model_dm([A, x], training=False)
        rho_gen = tf_to_dm(gen_dm)[0]
        states_list.append(rho_gen)
        f = fidelity(rho, states_list[-1])
        fidelity_list.append(f)
        pbar.set_description("F GAN {:.6f} Skip {} current_mean{:.6f}".format(f, skip_count, current_mean))
        # pbar.set_description("F: {:.2f} G lr = {:e} D lr = {:e}".format(f,
        #     generator_optimizer._decayed_lr(tf.float32).numpy(),
        #     discriminator_optimizer._decayed_lr(tf.float32).numpy()))
        if i > log_interval:
            current_mean = np.mean(fidelity_list[-log_interval:])

        if i > 2*log_interval:
            mfid_last_100 = np.mean(fidelity_list[-2*log_interval:-log_interval])

            if np.abs(mfid_last_100 - current_mean) < tol:
                skip_count += 1
                current_mean = mfid_last_100

        if skip_count > patience:
            pbar.close()
            break

    return fidelity_list, model_dm, states_list





def fit(x, loss, rho, ops_batch, max_iterations = 1000, noise=0, lr=None, generator=None,
    fidelity_list=None, states_list = None, generator_optimizer=None,
    log_interval=100,
              patience=5, tol=1e-4):
    hilbert_size = rho.shape[0]
    num_points = x.shape[1]

    tf.keras.backend.clear_session()
            
    if generator is None:
        generator = GeneratorSimple(hilbert_size, num_points, ops_batch, noise=noise)


    if lr == None:
        initial_learning_rate = 0.0002
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                             decay_steps=1000,
                                                             decay_rate=0.96,
                                                             staircase=False)
    else:
        lr_schedule = lr

    if generator_optimizer == None:
        generator_optimizer = tf.keras.optimizers.Adam(lr_schedule, 0.5, 0.5)

    def get_dm(generator):

        for i, layer in enumerate(generator.layers): 
                if "density_matrix" in layer._name:
                    density_layer_idx = i
                    break

        model_dm = tf.keras.Model(inputs=generator.input,
                                  outputs=generator.layers[density_layer_idx].output)
        return model_dm


    @tf.function()
    def train_step(x):
        with tf.GradientTape() as gen_tape:
            gen_output = generator(x, training=True)
            gen_loss = loss(x, gen_output)

        generator_gradients = gen_tape.gradient(gen_loss,
                                              generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))
        return gen_loss



    model_dm = get_dm(generator)

    
    gen_dm = model_dm(x, training=False)
    rho_gen = tf_to_dm(gen_dm)

    if states_list == None:
        states_list = [rho_gen[0]]
    else:
        states_list.append(rho_gen[0])

    if fidelity_list == None:
        fidelity_list = []
        fidelity_list.append(fidelity(rho, states_list[-1]))
    else:
        fidelity_list.append(fidelity(rho, states_list[-1]))

    pbar = tqdm(range(max_iterations))

    
    def train_step(x):
        with tf.GradientTape() as gen_tape:
            gen_output = generator(x, training=True)
            gen_loss = loss(x, gen_output)

        generator_gradients = gen_tape.gradient(gen_loss,
                                              generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))
        return gen_loss

    skip_count = 0
    current_mean = 0

    for i in pbar:
        train_step(x)
        gen_dm = model_dm(x, training=False)
        rho_gen = tf_to_dm(gen_dm)[0]
        states_list.append(rho_gen)

        f = fidelity(rho, states_list[-1])
        fidelity_list.append(f)
        pbar.set_description("F: {:.2f} Skip {} current_mean{:.6f}".format(f, skip_count, current_mean))
        pbar.update()
        # pbar.set_description("Fidelity {:.2f}; Generator lr = {:e}".format(f,
        #     generator_optimizer._decayed_lr(tf.float32).numpy()))
        if i > log_interval:
            current_mean = np.mean(fidelity_list[-log_interval:])

        if i > 2*log_interval:
            mfid_last_100 = np.mean(fidelity_list[-2*log_interval:-log_interval])

            if np.abs(mfid_last_100 - current_mean) < tol:
                skip_count += 1
                current_mean = mfid_last_100

        if skip_count > patience:
            pbar.close()
            break
    return fidelity_list, generator, states_list






