import numpy as np


from tqdm.auto import tqdm


from qutip import coherent_dm, destroy, mesolve, Options, fock_dm, coherent, expect
from qutip.visualization import plot_wigner_fock_distribution, plot_fock_distribution
from qutip.wigner import qfunc, wigner
from qutip import Qobj, qeye
from qutip.states import enr_state_dictionaries
from qutip.superoperator import liouvillian, spre, spost
from qutip import liouvillian, mat2vec, state_number_enumerate
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.solver import Options, Result, Stats
from qutip import mesolve


from scipy.special import factorial
from scipy.sparse import lil_matrix, csr_matrix
from scipy.integrate import ode


import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from qst_cgan.ops import cat, binomial, num, gkp
np.random.seed(42)


def add_photon_noise(rho0, gamma, tlist):
    """
    """
    n = rho0.shape[0]
    a = destroy(n)
    c_ops = [gamma*a,]
    H = -0*(a.dag() + a)
    opts = Options(atol=1e-20, store_states=True, nsteps=1500)
    L = liouvillian(H, c_ops=c_ops)
    states = mesolve(H, rho0, tlist, c_ops=c_ops)

    return states.states



def solve(L, rho0, tlist, options=None, e=0.8):
        """
        Solve the Lindblad equation given initial
        density matrix and time.
        """
        if options is None:
            options = Options()

        states = []
        states.append(rho0)
        
        n = rho0.shape[0]
        a = destroy(n)

        mean_photon_number = expect(a.dag()*a, rho0)

        dt = np.diff(tlist)
        rho = rho0.full().ravel("F")
        rho = rho.flatten()

        L = csr_matrix(L.full())
        r = ode(cy_ode_rhs)
        
        r.set_f_params(L.data, L.indices, L.indptr)

        r.set_integrator(
            "zvode",
            method=options.method,
            order=options.order,
            atol=options.atol,
            rtol=options.rtol,
            nsteps=options.nsteps,
            first_step=options.first_step,
            min_step=options.min_step,
            max_step=options.max_step,
        )

        r.set_initial_value(rho, tlist[0])

        n_tsteps = len(tlist)


        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                r1 = r.y.reshape((n, n))
                r1_q = Qobj(r1)
                states.append(r1_q)

                mphoton_number = np.real(expect(a.dag()*a, r1_q))

                if mphoton_number < e*mean_photon_number:
                    break

        return states
