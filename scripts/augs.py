import numpy as np


# DANS: Data Augmentations for Nuclear Spectra feature-Extraction
class DANSE:
    def __init__(self):
        pass

    def background(self, X, X_bckg, subtraction=False,
                   event_idx=None, mode='mean'):
        '''
        Superimposes an even signature onto various forms of background
        distributions. This action does require an accurate estimation of a
        typical baseline, but several methods are employed.
        X is assumed to be background adjusted already. That is, it should
        have its original background already removed.
        If subtraction=True and event_idx is not None, X should be 2D.
        event_idx indicates the row in X that indicates the event spectrum.
        The other rows in X are then used to estimate a background distribution
        for subtraction.
        NOTE: Two background subtraction modes are supported: 'min' and 'mean'.
        'min': take the minimum gross count-rate spectrum from X.
        'mean': take the average count-rate for each bin from X.

        Inputs:
        X: array-like; If 1D, taken as an event spectrum (must be previously
            background-subtracted). If 2D, subtraction=True, and event_idx not
            None, a background subtraction is conducted prior to
            superimposition.
        X_bckg: array-like; If 1D, add this spectrum to the event spectrum of X
            as the superimposed background. If 2D, use the mode input to
            complete background superimposition.
        subtraction: bool; If True, conduct background subtraction on X
            (event_idx must not be None)
        event_idx: int(p); row index for event spectrum in X used for
            background subtraction.
        mode: str; two background subtraction modes are currently supported:
            'min': take the minimum gross count-rate spectrum from X.
            'mean': take the average count-rate for each bin from X.
        '''

        modes = ['min', 'mean']
        # input error checks
        if subtraction and event_idx is None:
            raise ValueError('If subtraction=True,',
                             'event_idx must be specified.')
        elif subtraction and event_idx is not None and len(X) <= 1:
            raise ValueError('X must be 2D to do background subtraction.')
        elif mode not in modes:
            raise ValueError('Input mode not recognized.')

        # subtract a background estimation if it wasn't done prior
        if subtraction:
            bckg = np.delete(X, event_idx, axis=0)
            X = X[event_idx]
            if mode == 'min':
                idx = np.argmin(np.sum(bckg, axis=0))
                X -= bckg[idx]
            elif mode == 'mean':
                X -= np.mean(bckg, axis=1)

        # estimate a background/baseline if multiple spectra are provided
        if len(X_bckg) > 1:
            if mode == 'min':
                idx = np.argmin(np.sum(X_bckg, axis=0))
                X_bckg = X_bckg[idx]
            elif mode == 'mean':
                X_bckg = np.mean(X_bckg, axis=1)

        return X + X_bckg

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
