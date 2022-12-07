import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.signal import find_peaks


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
            raise ValueError('Input mode not supported.')

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
            raise ValueError('If subtraction=True, ',
                             'event_idx must be specified.')
        elif subtraction and event_idx is not None and len(X) <= 1:
            raise ValueError('X must be 2D to do background subtraction.')
        elif mode not in modes:
            raise ValueError('Input mode not supported.')

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

    def _gauss(self, x, amp, mu, sigma):
        '''
        Fit equation for a Gaussian distribution.
        Inputs:
        x: array-like; 1D spectrum array of count-rates
        amp: float; amplitude = A/sigma*sqrt(2*pi)
        mu: float; mean
        sigma: float; standard deviation
        '''

        return amp * np.exp(-((x - mu) / 4 / sigma)**2)

    def _emg(self, x, amp, mu, sigma, tau):
        """
        Exponentially Modifed Gaussian (for small tau). See:
        https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
        Inputs:
        x: array-like; 1D spectrum array of count-rates
        amp: float; amplitude = A/sigma*sqrt(2*pi)
        mu: float; mean
        sigma: float; standard deviation
        tau: float; exponent relaxation time
        """

        term1 = np.exp(-0.5 * np.power((x - mu) / sigma, 2))
        term2 = 1 + (((x - mu) * tau) / sigma**2)
        return amp * term1 / term2

    def _lingauss(self, x, amp, mu, sigma, m, b):
        '''
        Includes a linear term to the above function. Used for modeling
        (assumption) linear background on either shoulder of a gamma photopeak.
        Inputs:
        x: array-like; 1D spectrum array of count-rates
        amp: float; amplitude = A/sigma*sqrt(2*pi)
        mu: float; mean
        sigma: float; standard deviation
        m: float; linear slope for background/baseline
        b: float; y-intercept for background/baseline
        '''

        return amp * np.exp(-0.5 * np.power((x - mu) / sigma, 2.)) + m*x + b

    def _fit(self, roi, X):
        '''
        Fit function used by resolution() for fitting a Gaussian function
        on top of a linear background in a specified region of interest.
        TODO: Add a threshold for fit 'goodness.' Return -1 if failed.
        Inputs:
        roi: tuple; (min, max) bin/index values for region of interest - used
            to index from data, X
        X: array-like; 1D spectrum array of count-rates
        '''

        # binning of data (default usually 0->1000 bins)
        ch = np.arange(0, len(X))
        region = X[roi[0]:roi[1]]

        # initial guess for fit
        max_y = np.max(region)
        max_z = ch[roi[0]:roi[1]][np.argmax(region)]
        # [amp, mu, sigma, m, b]
        p0 = [max_y, max_z, 1., 0, X[roi[0]]]

        coeff, var_matrix = curve_fit(self._lingauss,
                                      ch[roi[0]:roi[1]],
                                      region,
                                      p0=p0)

        return coeff

        # # as calculated exactly from Gaussian statistics
        # fwhm = 2*np.sqrt(2*np.log(2))*coeff[1]
        # return fwhm

    def _crude_bckg(self, roi, X):
        '''
        Linear estimation of background using the bounds of an ROI.
        Uses point-slope formula and the bounds for the ROI region to create
        an array of the expected background.
        Inputs:
        roi: tuple; (min, max) bin/index values for region of interest - used
            to index from data, X
        X: array-like; 1D spectrum array of count-rates
        '''

        lower_bound = roi[0]
        upper_bound = roi[1]

        y1 = X[lower_bound]
        y2 = X[upper_bound]
        slope = (y2 - y1) / (upper_bound - lower_bound)

        y = slope * (np.arange(lower_bound, upper_bound) - lower_bound) + y1

        return y, slope, y1

    def nuclear(self, X, E, binE=3., sigma=0., escape=False):
        # escape peak error to ensure physics
        if escape and E < 1022:
            raise ValueError('Photopeaks below 1,022 keV ',
                             'do not produce escape peaks.')
        # avoid overwriting original data
        X = X.copy()
        bins = X.shape[0]

        # find (photo)peaks with heights above baseline of at least 100 counts
        peaks, properties = find_peaks(X, prominence=100)
        # find the two tallest peak to estimate energy resolution
        fit_peaks = peaks[np.argsort(properties['prominences'])[-2:]]
        # fit the two most prominent peaks
        # [amp, mu, sigma, m, b]
        coeff1 = self._fit([fit_peaks[0]-10, fit_peaks[0]+10], X)
        amp1, sigma1 = coeff1[0], coeff1[2]
        coeff2 = self._fit([fit_peaks[1]-10, fit_peaks[1]+10], X)
        amp2, sigma2 = coeff2[0], coeff2[2]
        # assume linear relationship in peak counts and width over spectrum
        # TODO: add user input for peak intensity/counts/amplitude
        slope_sigma = abs((sigma2 - sigma1)/(fit_peaks[1] - fit_peaks[0]))
        slope_counts = np.sqrt(2*np.pi) * abs((amp2 - amp1) /
                                              (fit_peaks[1] - fit_peaks[0]))

        # calculate bin for input energy
        b = int(E/binE)
        # insert peak at input energy
        if not escape:
            # approximate width and counts from relationship estimated above
            sigma_peak = slope_sigma * b
            size_peak = slope_counts * b
            # create another spectrum with only the peak
            new_peak, _ = np.histogram(np.round(
                                        np.random.normal(loc=b,
                                                         scale=sigma_peak,
                                                         size=int(size_peak))),
                                       bins=bins,
                                       range=(0, bins))
            X = X+new_peak
        # insert escape peaks if specified or physically realistic
        if escape or E >= 1022:
            # fit the peak at input energy
            # [amp, mu, sigma, m, b]
            coeff = self._fit([b-10, b+10], X)
            # background counts integral
            width = (b+10)-(b-10)
            background = (coeff[3]/2)*((b+10)**2
                                       - (b-10)**2) + coeff[4] * (width)
            # find difference from background
            peak_counts = np.sum(X[b-10:b+10]) - background
            print(peak_counts, background)

            # normal distribution parameters for single and double escape peaks
            b_single = int((E-511)/binE)
            sigma_single = slope_sigma * b_single
            size_single = ((E-511)/E)*peak_counts
            b_double = int((E-1022)/binE)
            sigma_double = slope_sigma * b_double
            size_double = ((E-1022)/E)*peak_counts

            # create another spectrum with only the peak for each escape peak
            single, _ = np.histogram(np.round(
                                      np.random.normal(loc=b_single,
                                                       scale=sigma_single,
                                                       size=int(size_single))),
                                     bins=bins,
                                     range=(0, bins))
            double, _ = np.histogram(np.round(
                                      np.random.normal(loc=b_double,
                                                       scale=sigma_double,
                                                       size=int(size_double))),
                                     bins=bins,
                                     range=(0, bins))
            X = X+single+double
        return X

    def resolution(self, roi, X, multiplier=1.5):
        # avoid overwriting original data
        X = X.copy()

        # [amp, mu, sigma, m, b]
        coeff = self._fit(roi, X)
        fwhm = 2*np.sqrt(2*np.log(2))*coeff[2]
        new_sigma = multiplier * fwhm / (2*np.sqrt(2*np.log(2)))
        coeff[2] = new_sigma

        # there's no need to refind background/baseline
        # because it was fit in coeff above
        # but this could be used to isolate background
        # y, m, b = self._crude_bckg(roi, X)

        ch = np.arange(roi[0], roi[1])
        peak = self._lingauss(ch,
                              amp=coeff[0],
                              mu=coeff[1],
                              sigma=new_sigma,
                              m=coeff[3],
                              b=coeff[4])

        # add noise to the otherwise smooth transformation
        peak = self.resample(peak)
        X[roi[0]:roi[1]] = peak
        return X

    def mask(self, X, mode='random', interval=5, block=(0, 100)):
        '''
        Mask specific regions of a spectrum to force feature importance.
        This may or may not be physically realistic, depending on the masking
        scenario (e.g. pileup) but it represents a common image augmentation.
        NOTE: the default values for interval and block are not used, but
        recommended sizes or degrees for reasonable augmentations.

        Inputs:
        X: array-like; should be 1D, i.e. one spectrum to be augmented
        mode: str; three modes are supported:
            'interval': mask every interval's channel
            'block': mask everything within a block range
            'both': mask every interval's channel within a block range
            'random': randomly pick one of the above
        interval: int; mask every [this int] channel in the spectrum
        block: tuple; spectral range to mask (assumed spectral length is
            1000 channels)
        '''

        modes = ['interval', 'block', 'both']
        if mode != 'random' or mode not in modes:
            raise ValueError('Input mode not supported.')
        if mode == 'random':
            mode = np.random.choice(modes)
            if mode == 'interval':
                # high => exclusive: 10+1
                interval = np.random.randint(1, 11)
            elif mode == 'block':
                # default spectral length is 1,000 channels
                # TODO: abstract spectral length
                low = np.random.randint(0, 999)
                # default block width is low+10 to max length
                # TODO: abstract block width
                high = np.random.randint(low+10, 1000)
                block = (low, high)

        # mask spectrum (i.e. set values to 0)
        if mode == 'interval':
            X[::interval] = 0
        elif mode == 'block':
            X[block[0]:block[1]] = 0
        elif mode == 'both':
            X[block[0]:block[1]:interval] = 0

        return X

    def ResampleLinear1D(original, targetLen):
        '''
        https://stackoverflow.com/questions/20322079/downsample-a-1d-numpy-array
        '''
        original = np.array(original, dtype=float)
        index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=float)
        index_floor = np.array(index_arr, dtype=int)  # Round down
        index_ceil = index_floor + 1
        index_rem = index_arr - index_floor  # Remain

        val1 = original[index_floor]
        val2 = original[index_ceil % len(original)]
        interp = val1 * (1.0-index_rem) + val2 * index_rem
        assert(len(interp) == targetLen)
        return interp

    def _ResampleLinear1D(self, original, targetLen):
        '''
        Originally from StackOverflow.
        Upsamples or downsamples an array by interpolating
        the value in each bin to a given length.

        Inputs:
        original: array-like; spectrum or array to be resampled
        targetLen: int; target length to resize/resample array
        '''

        original = np.array(original, dtype=float)
        index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=float)
        # find the floor (round-down) for each bin (cutting off with int)
        index_floor = np.array(index_arr, dtype=int)
        # find the ceiling (max/round-up) for each bin
        index_ceil = index_floor + 1
        # compute the difference/remainder
        index_rem = index_arr - index_floor

        val1 = original[index_floor]
        val2 = original[index_ceil % len(original)]
        # interpolate the new value for each new bin
        interp = val1 * (1.0-index_rem) + val2 * index_rem
        assert(len(interp) == targetLen)
        return interp

    def gain_shift(self, counts, mu=np.random.uniform(0, 5),
                   k=0, bins=None, negative=False):
        '''
        Modulate the gain-shift underlying a spectrum.
        This simulates a change in the voltage to channel mapping, which
        will affect how the spectral shape appears in channel vs. energy space.
        If a positive gain shift occurs (multiplier increases), e.g. 1V=1ch
        becomes 0.9V=1ch, spectral features will stretch out and widen across
        the spectrum. Vice versa for a negative gain shift.

        Inputs:
        counts: array-like; 1D spectrum, with count-rate for each channel
        mu: float; Poisson parameter for gain drift. Determines the severity
            of gain drift in spectrum. As of right now, the drift is energy
            dependent (i.e. more drift for higher energies).
        k: int; number of bins to shift the entire spectrum by
        bins: array-like; 1D vector (with length len(counts)+1) of either
            bin edges in energy space or channel numbers.
        negative: bool; determine whether gain shift/drift is in a negative
            direction instead of the default positive.
            Can also be used instead of positive shift algorithm by using the
            combination negative=True, mu<0
            NOTE: Which algorithm should be kept, both?
        '''

        if len(counts.shape) > 1:
            raise ValueError(f'gain_shift expects only 1 spectrum (i.e. 1D \
                               vector) but {counts.shape[0]} were passed')

        # gain-shift algorithm
        # add blank bins before or after the spectrum
        if k < 0:
            counts = np.append(counts, np.repeat(0., np.absolute(k)))
            counts[0] = np.sum(counts[:np.absolute(k)])
            counts = np.delete(counts, np.arange(1, np.absolute(k)))
            # fix the length of the spectrum to be the same as before
            if bins is not None:
                bins = np.linspace(bins[0], bins[-1], counts.shape[0]+1)
        elif k > 0:
            counts = np.insert(counts, 0, np.repeat(0., k))
            # fix the length of the spectrum to be the same as before
            if bins is not None:
                width = bins[1] - bins[0]
                bins = np.arange(bins[0], bins[-1]+(k*width), width)

        # negative shift using downsampling
        if negative:
            new_b = bins
            new_ct = self._ResampleLinear1D(counts, int(counts.shape[0]+mu))
            # enforce the same count-rate
            new_ct *= np.sum(counts)/np.sum(new_ct)
            if bins is not None:
                width = bins[1] - bins[0]
                new_b = np.arange(bins[0],
                                  bins[0]+((len(new_ct)+1)*width),
                                  width)
            return new_ct, new_b

        if mu is None:
            return counts, bins

        # gain-drift algorithm
        new_ct = counts.copy()
        for i, c in enumerate(counts):
            # randomly sample a new assigned index for every count in bin
            # using np.unique, summarize which index each count goes to
            idx, nc = np.unique(np.round(poisson.rvs(mu=mu*(i/counts.shape[0]),
                                                     size=int(c))),
                                return_counts=True)
            # check to see if any indices are greater than the spectral length
            missing_idx = np.count_nonzero(i+idx >= new_ct.shape[0])
            if missing_idx > 0:
                # add blank bins if so
                new_ct = np.append(new_ct,
                                   np.repeat(0,
                                             np.max(idx)+i-new_ct.shape[0]+1))
            # distribute all counts according to their poisson index
            new_ct[(i+idx).astype(int)] += nc
            # adjust for double-counting
            new_ct[i] -= np.sum(nc)
        # recalculate binning if passed
        new_b = bins
        if bins is not None:
            width = bins[1] - bins[0]
            new_b = np.arange(bins[0], bins[0]+((len(new_ct)+1)*width), width)

        return new_ct, new_b
