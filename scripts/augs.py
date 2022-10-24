import numpy as np


# DANS: Data Augmentations for Nuclear Spectra feature-Extraction
# TODO: standardize return to either include background or not
class DANSE:
    def __init__(self):
        pass

    def _estimate(self, X_bckg, mode):
        '''
        Background estimation method used in background and sig2bckg.
        NOTE: Two background subtraction modes are supported: 'min' and 'mean'.
        'min': take the minimum gross count-rate spectrum from X.
        'mean': take the average count-rate for each bin from X.

        Inputs:
        X_bckg: array-like; 2D spectral array of measurements, uses the mode
            input to complete background superimposition.
        mode: str; two background subtraction modes are currently supported:
            'min': take the minimum gross count-rate spectrum from X.
            'mean': take the average count-rate for each bin from X.
        '''

        if mode == 'min':
            idx = np.argmin(np.sum(X_bckg, axis=0))
            X_bckg = X_bckg[idx]
        elif mode == 'mean':
            X_bckg = np.mean(X_bckg, axis=1)

        return X_bckg

    def background(self, X, X_bckg, subtraction=False,
                   event_idx=None, mode='mean'):
        '''
        Superimposes an even signature onto various forms of background
        distributions. This action does require an accurate estimation of a
        typical baseline, but several methods are employed.
        X is assumed to be background adjusted by default. That is, it should
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

            bckg = self.estimate(bckg, mode)
            X -= bckg

        # estimate a background/baseline if multiple spectra are provided
        if len(X_bckg) > 1:
            X_bckg = self.estimate(X_bckg, mode)

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

    def sig2bckg(self, X, X_bckg, r=(0.5, 2.), subtraction=False,
                 event_idx=None, mode='mean'):
        '''
        Estimate and subtract background and scale signal-to-noise of event
        signature. The return is a spectrum with an estimated background and
        a perturbed signal intensity.
        Scaling ratio is 1/r^2. Therefore, r<1 makes the signal more intense
        and r>1 makes the signal smaller.
        X is assumed to be background adjusted by default. That is, it should
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
        r: tuple; [min, max) scaling ratio. Default values ensure random
            scaling that is no more than 4x larger or smaller than the original
            signal. See numpy.random.uniform for information on interval.
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

            bckg = self.estimate(bckg, mode)
            X -= bckg

        # estimate a background/baseline if multiple spectra are provided
        if len(X_bckg) > 1:
            X_bckg = self.estimate(X_bckg, mode)

        r = np.random.uniform(r[0], r[1])

        X *= 1/r**2

        return X + X_bckg

    def nuclear(self):
        pass

    def resolution(self):
        pass

    def mask(self):
        pass

    def gain_shift(self):
        pass
