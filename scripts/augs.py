import numpy as np


# DANS: Data Augmentations for Nuclear Spectra
class DANS:
    def __init__(self):
        pass

    def background(self):
        pass

    def resample(self, X):
        '''
        Resamples spectra according to a Poisson distribution.
        Gamma radiation detection is approximately Poissonian.
        Each energy bin of a spectrum could be resampled using the original
        count-rate, lambda_i, as the statistical parameter for a distribution:
        Pois_i(lambda_i). Randomly sampling from this distribution would
        provide a new count-rate for that energy bin that is influenced, or
        augmented, by the original sample.

        Inputs:
        X: array-like; can be a vector of one spectrum, a matrix of many
            matrices (rows: spectra, cols: instances), or a subset of either.
            X serves as the statistical parameters for each distribution.
        Return:
        augmentation: array-like, same shape as X; the augmented spectra using
            channel resampling (see above)
        '''

        augmentation = np.random.poisson(lam=X)

        return augmentation

    def sig2bckg(self):
        pass

    def nuclear(self):
        pass

    def resolution(self):
        pass

    def mask(self):
        pass

    def gain_shift(self):
        pass
