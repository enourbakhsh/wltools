import numpy as np
import matplotlib.pyplot as plt


class massfunction(object):
    def __init__(self, opt):
        types = ['st', 'jenkins', 'tinker']
        if opt not in types:
            print("opt must be one of ('st', 'jenkins', 'tinker')")
            self.type = None
        else:
            self.type = opt

    def fsig(self, sig):
        # """Returns comoving number density of halos with mass M [M_sun]."""
        # Assumes M is in units of M_sun
        # TODO: verify M units
        if self.type == 'st':
            A = 0.3222
            q = 0.707
            p = 0.3
            delta_c = 1.686
            fsig = (A * np.sqrt(2 * q / np.pi) * (delta_c / sig) *
                    (1 + (sig**2 / q / delta_c**2)**p) *
                    np.exp(-q * delta_c**2 / 2 / sig**2))
        elif self.type == 'jenkins':
            A = 0.315
            b = 0.61
            c = 3.8
            fsig = A * np.exp(-abs(np.log(1 / sig) + b)**c)
        elif self.type == 'tinker':
            # TODO
            fsig = 0
        else:
            print("Invalid type. Returning 0.")
            return 0

        return fsig

    def plot_fsig(self):
        sig = np.linspace(0.0001, 10, 256)
        fig, ax = plt.subplots(1, 1, facecolor='w')
        ax.plot(sig, self.fsig(sig), label=self.type)
        ax.set_xlabel(r'$\sigma$')
        ax.set_ylabel(r'$f(\sigma)$')
        plt.legend(loc=0)
        plt.show()
