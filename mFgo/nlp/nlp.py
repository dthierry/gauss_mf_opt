# -*- coding: utf-8 -*-

import numpy as np
import cyipopt
import logging
from scipy.stats import qmc
from mFgo.mfm.MultiFidelity import MultiFidelity
from mFgo.plt.pG import genplot
import matplotlib.pyplot as plt

__author__ = "David Thierry @dthierry"

def f(x):
    """Toy problem"""
    return np.stack([
           3/2-x[:,0]-2*x[:,1]-1/2*np.sin(2*np.pi*(x[:,0]**2-2*x[:,1])),
           x[:,0]**2+x[:,1]**2-3/2\
                    ],-1) # y1, y2


class mfG(object):
    def __init__(self):
        """This basically trains the model."""
        sampler = qmc.Sobol(d=2, scramble=True)
        #xlf_np = sampler.random_base2(m=10)[:pts_lf, :]
        pts_lf = 100
        pts_t = 100

        xlf_np = sampler.random_base2(m=10)[:pts_lf,:] # Training x-samples
        ylf_np = f(xlf_np) # sampws
        num_tasks = ylf_np.shape[1]
        num_inputs = xlf_np.shape[1]
        max_epochs = 200
        early_stopping_tol = [200, 1e-03]
        lrs = [1e-02, 1e-06]
        restarts = [0, 400]
        self.mfG = MultiFidelity(num_inputs,
                                 num_tasks,
                                 deep=False)
        self.mfG.train(xlf_np,
                       ylf_np,
                       fidelity=1,
                       epochs=max_epochs,
                       verbose=True,
                       early_stopping_tol=early_stopping_tol,
                       lrs=lrs,
                       restarts=restarts)
        logging.info("the model has been trained!!!")


class gauss_reg_nlp:
    def __init__(self):
        """Load the regression"""
        self.mfGm = mfG()
        self.xp = np.array([])

    def objective(self,x):
        return x[0] + x[1]

    def gradient(self, x):
        """objective gradient"""
        # of the objective or whaat?
        return np.array([1, 1])

    def constraints(self, x):
        """constraints, i.e. the gaussian proc regression"""
        fstar, _, _, _ = self.mfGm.mfG.test(x.reshape(1,-1), fidelity=1)
        fstar = fstar.reshape(2,)
        return fstar

    def jacobian(self, x):
        """Jacobian of the constraints"""
        if self.xp.size == 0:
            self.xp = x
        else:
            self.xp = np.vstack((self.xp, x))

        j0 = self.mfGm.mfG.Jacobian(x.reshape(1,-1), fidelity=1)
        return j0.reshape(4,)

    def hessian(self, x, lagrange, obj_factor):
        """We don't have second derivatives yet."""
        H = np.zeros((2,2))  #: the objective is linear :(
        return np.concatenate(H)

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm, regularization_size,
                     alpha_du, alpha_pr, ls_trials):
        msg = "henlo iteration={:d} obj={:g}"
        #print(msg.format(iter_count, obj_value))

# bounds
lb = [0e0, 0e0]
ub = [1e0, 1e0]
# constraint bounds
cl = [-2e19, -2e19]
cu = [0e0, 0e0]
# initial point
x0 = [0.5, 0.5]


# problem object
po=gauss_reg_nlp()
# nlp object
nlp = cyipopt.Problem(n=2, # vars
                      m=2,  # cons
                      problem_obj=po, # prob object
                      lb=lb, ub=ub,
                      cl=cl, cu=cu)

# we need these!
nlp.add_option("hessian_approximation", "limited-memory")
nlp.add_option("limited_memory_update_type", "bfgs")


#plots
fig, ax = genplot(po.mfGm.mfG)

# Solve!!
nlp.solve(x0)

xsoln = np.hsplit(po.xp, 2)

# this is supposed to plot the path:S
ax.plot(xsoln[0], xsoln[1], "o", label="opt", color="red")

ax.legend()
fig.savefig("opt.png", dpi=300)

