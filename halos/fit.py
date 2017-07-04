from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import emcee
from ..simulations import nfw_simulation


class mcmcfit(object):
    """Fit a halo profile to shear data. Only NFW is currently supported."""
    def __init__(self):
        pass

    def generate_data(self):
        # NFW halo at (0, 0) in a 5x5 arcmin^2 field
        sim = nfw_simulation(x0=0, y0=0, r200=1, c=5, z=0.4, mode='grid')
        self.x_gal = sim.ra_gal
        self.y_gal = sim.dec_gal
        self.e1 = sim.gamma1
        self.e2 = sim.gamma2
        self.z_gal = sim.z_gal

    def load_catalog(self, x_gal, y_gal, e1, e2, z_gal):
        self.x_gal = x
        self.y_gal = y
        self.e1 = e1
        self.e2 = e2
        self.z_gal = z_gal

    def run_mcmc(self):
        pass


class leastsquaresfit(object):
    pass


class mcmctest(object):
    def __init__(self):
        # Set true model parameters
        self.m_true = -0.9594
        self.b_true = 4.294
        self.f_true = 0.534

        # Generate data
        N = 50
        x = np.sort(10 * np.random.rand(N))
        y = self.m_true * x + self.b_true
        # Add underestimated uncertainties
        y += np.abs(self.f_true * y) * np.random.randn(N)
        yerr = 0.1 + 0.5 * np.random.rand(N)
        y += yerr * np.random.randn(N)

        # Maximize likelihood
        nll = lambda *args: -self.lnlike(*args)
        p0 = [self.m_true, self.b_true, np.log(self.f_true)]
        result = opt.minimize(nll, p0, args=(x, y, yerr))
        self.m_ml, self.b_ml, self.lnf_ml = result['x']

        ndim, nwalkers = 3, 100
        pos = [result['x'] + (1e-4) * np.random.randn(ndim) for
               i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                             args=(x, y, yerr))
        sampler.run_mcmc(pos, 500)
        # self.chain = sampler.chain
        self.m_chain = sampler.chain[:, :, 0]
        self.b_chain = sampler.chain[:, :, 1]
        self.f_chain = sampler.chain[:, :, 2]
        self.lnprobability = sampler.lnprobability

        samples = sampler.chain[:, 50:, :].reshape((-1, ndim)).T
        # Return parameter lnf to f
        samples[2] = np.exp(samples[2])
        self.samples = samples
        self.m_mcmc, self.b_mcmc, self.f_mcmc = map(lambda v:
            (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(self.samples, [16, 50, 84], axis=1)))

        fig, ax = plt.subplots(1, 1, facecolor='w')
        ax.plot(x, self.m_true * x + self.b_true, c='k', label='truth')
        ax.errorbar(x, y, yerr, fmt='.k', ms=5, capsize=2)
        ax.plot(x, self.m_ml * x + self.b_ml, c='b', linestyle='dashed',
                label='max likelihood')
        ax.plot(x, self.m_mcmc[0] * x + self.b_mcmc[0], c='r', label='mcmc')
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.legend()
        plt.show()

    def lnlike(self, theta, x, y, yerr):
        m, b, lnf = theta
        model = m * x + b
        sigma2 = yerr**2 + np.exp(2 * lnf) * model**2
        return -0.5 * (np.sum((y - model)**2 / sigma2 + np.log(sigma2)))

    def lnprior(self, theta):
        # Check that parameter set is within prior limits
        m, b, lnf = theta
        if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
            return 0.0
        return -np.inf

    def lnprob(self, theta, x, y, yerr):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, x, y, yerr)
