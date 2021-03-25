import numpy as np
from qst_cgan.ops import convert_to_complex_ops

from qutip import Qobj, expect, fidelity, qeye
from qutip.states import thermal_dm
from qutip.random_objects import rand_dm

from tqdm.auto import tqdm

def train_imle(A, x, rho_true,
               max_iterations=100,
               states_list=None,
               fidelity_list=None,
               log_interval=100,
               patience=10, tol=1e-3):

    np_ops = convert_to_complex_ops(A)[0].numpy()
    qutip_ops = [Qobj(op) for op in np_ops]

    np_ops = np.array([op.full() for op in qutip_ops])
    
    hilbert_size = A.shape[1]
    num_points = int(A.shape[-1]/2)
    
    data = x.numpy().ravel()
    rho = rand_dm(hilbert_size, 0.8)
    # rho = rho.unit()
    
    if fidelity_list == None:
        fidelity_list = []

    f = fidelity(rho_true, rho)
    fidelity_list.append(f)
    
    if states_list == None:
        states_list = []

    states_list.append(rho)

    pbar = tqdm(range(max_iterations))

    skip_count = 0
    current_mean = 0

    for i in range(max_iterations):
        guessed_val = expect(qutip_ops, rho)
        ratio = data/guessed_val

        R = Qobj(np.einsum("aij,a->ij", np_ops, ratio))

        rho = R*rho*R
        try:
            rho = rho/rho.tr()
        except:
            pass
        states_list.append(rho)

        f = fidelity(rho_true, rho)
        fidelity_list.append(f)

        pbar.set_description("F iMLE {:.4f} Skip {} current_mean{:.4f}".format(f, skip_count, current_mean))
        pbar.update()

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

    return fidelity_list, None, states_list