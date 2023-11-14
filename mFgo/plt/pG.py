# -*- coding: utf-8 -*-

import numpy as np # NumPy
from matplotlib import pyplot as plt # MATLAB Plot Utilities
from matplotlib import patheffects
from matplotlib.colors import Normalize as nrm
import logging
from scipy.stats import qmc
from mFgo.mfm.MultiFidelity import MultiFidelity

__author__ = "David J Thierry"

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


def objf(x0, x1):
    return x0 + x1
def c0(x0, x1):
    return 3/2-x0-2*x1-1/2*np.sin(2*np.pi*(x0**2-2*x1))
def c1(x0, x1):
    return x0**2+x1**2-3/2
def f(x):
    return np.stack([
           3/2-x[:,0]-2*x[:,1]-1/2*np.sin(2*np.pi*(x[:,0]**2-2*x[:,1])),
           x[:,0]**2+x[:,1]**2-3/2\
                    ],-1) # y1, y2



def genplot(mfG):
    sampler = qmc.Sobol(d=2, scramble=True)
    #xlf_np = sampler.random_base2(m=10)[:pts_lf, :]
    pts_lf = 100
    pts_t = 100

    xlf_np = sampler.random_base2(m=10)[:pts_lf,:] # Training x-samples
    ylf_np = f(xlf_np) # sampws
    num_tasks = ylf_np.shape[1]
    num_inputs = xlf_np.shape[1]
    max_epochs = 5000
    early_stopping_tol = [200, 1e-03]
    lrs = [1e-02, 1e-06]
    restarts = [0, 400]
    mfG = MultiFidelity(num_inputs,
                             num_tasks,
                             deep=False)
    mfG.train(xlf_np,
                   ylf_np,
                   fidelity=1,
                   epochs=max_epochs,
                   verbose=True,
                   early_stopping_tol=early_stopping_tol,
                   lrs=lrs,
                   restarts=restarts)
    logging.info("the model has been trained!!!")

    x0, x1 = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
    x = np.stack((x0, x1), axis=-1)

    of = objf(x0, x1)
    #y = f(x)
    i = 0
    for xk in x:
        #y = f(xk)
        y, _, _, _ = mfG.test(xk, fidelity=1)
        if i == 0:
            y0 = y[:,0].reshape(1,50)
            y1 = y[:,1].reshape(1,50)
        else:
            y0 = np.concatenate((y0, [y[:,0]]))
            y1 = np.concatenate((y1, [y[:,1]]))
        i+=1

    cmap = plt.get_cmap("inferno")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    cntr = ax.contour(x0, x1, of, [0.5, 0.75, 1.0, 1.25],
                  cmap=cmap, norm=nrm(vmin=0.1, vmax=1.5),
                  linestyles="dashdot"
                  #colors='black'
                  )
    ax.clabel(cntr, fmt=r"$f$=%2.1f", use_clabeltext=True)

    cg1 = ax.contour(x0, x1, y0, [0], colors='sandybrown')
    plt.setp(cg1.collections,
             path_effects=[patheffects.withTickedStroke(angle=60)])

    ax.clabel(cg1, fmt=r"GPS$_1(x)$=%1.f", use_clabeltext=True)
    cg2 = ax.contour(x0, x1, y1, [0], colors='orangered')
    plt.setp(cg2.collections,
             path_effects=[patheffects.withTickedStroke(angle=30, length=2)])
    ax.clabel(cg2, fmt=r"GPS$_2(x)$=%1.f", use_clabeltext=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_title("Gaussian Process Regression")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    x0 = [0.1954,0.7197,0]
    y0 = [0.4044,0.1411,0.75]
    ax.plot(x0, y0, "*", color="red", label="Local solutions")
    ax.legend()
    return fig, ax

if __name__ == "__main__":
    f, a = genplot(1)
    f.savefig("trained1.png")
