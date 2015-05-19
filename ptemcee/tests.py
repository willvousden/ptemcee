#!/usr/bin/env python
# encoding: utf-8
'''
Defines various nose unit tests

'''

from __future__ import division

import numpy as np
from .sampler import Sampler

logprecision = -4

def logprob_gaussian(x, icov):
    return -np.dot(x, np.dot(icov, x)) / 2.0

def logprob_gaussian_nan(x, icov):
    # if walker's parameters are zeros => return NaN
    if not (np.array(x)).any():
        result = np.nan
    else:
        result = -np.dot(x, np.dot(icov, x)) / 2.0

    return result

def log_unit_sphere_volume(ndim):
    if ndim % 2 == 0:
        logfactorial = 0.0
        for i in range(1, ndim / 2 + 1):
            logfactorial += np.log(i)
        return ndim / 2.0 * np.log(np.pi) - logfactorial
    else:
        logfactorial = 0.0
        for i in range(1, ndim + 1, 2):
            logfactorial += np.log(i)
        return (ndim + 1) / 2.0 * np.log(2.0) \
            + (ndim - 1) / 2.0 * np.log(np.pi) - logfactorial

class LogLikeGaussian(object):
    def __init__(self, icov, test_nan=False):
        '''Initialize a gaussian PDF with the given inverse covariance
        matrix.  If not ``None``, ``cutoff`` truncates the PDF at the
        given number of sigma from the origin (i.e. the PDF is
        non-zero only on an ellipse aligned with the principal axes of
        the distribution).  Without this cutoff, thermodynamic
        integration with a flat prior is logarithmically divergent.'''

        self.icov = icov
        self.test_nan = test_nan

    def __call__(self, x):
        f = logprob_gaussian_nan if self.test_nan else logprob_gaussian
        dist2 = f(x, self.icov)

        return dist2

class LogPriorGaussian(object):
    def __init__(self, icov, cutoff=None):
        self.icov = icov
        self.cutoff = cutoff

    def __call__(self, x):
        dist2 = logprob_gaussian(x, self.icov)

        if self.cutoff is not None:
            if -dist2 > self.cutoff * self.cutoff / 2.0:
                return float('-inf')
            else:
                return 0.0
        else:
            return 0.0

class Tests(object):

    def setUp(self):
        self.nwalkers = 100
        self.ndim = 5

        self.ntemps = 10
        self.Tmax = 250
        self.cutoff = 10

        self.N = 1000

        self.mean = np.zeros(self.ndim)
        self.cov = 0.5 - np.random.rand(self.ndim ** 2) \
            .reshape((self.ndim, self.ndim))
        self.cov = np.triu(self.cov)
        self.cov += self.cov.T - np.diag(self.cov.diagonal())
        self.cov = np.dot(self.cov, self.cov)
        self.icov = np.linalg.inv(self.cov)
        self.p0 = [[0.1 * np.random.randn(self.ndim)
                    for i in range(self.nwalkers)]
                   for j in range(self.ntemps)]

    def check_sampler(self, cutoff=None, N=None, p0=None, adapt=False):
        if cutoff is None:
            cutoff = self.cutoff
        if N is None:
            N = self.N
        if p0 is None:
            p0 = self.p0

        for i in self.sampler.sample(p0, iterations=N, adapt=adapt):
            pass

        # Weaker assertions on acceptance fraction
        assert np.mean(self.sampler.acceptance_fraction) > 0.1, \
            'acceptance fraction < 0.1'
        assert np.mean(self.sampler.tswap_acceptance_fraction) > 0.1, \
            'tswap acceptance fraction < 0.1'
        # TODO
        # assert abs(self.sampler.tswap_acceptance_fraction[0] - 0.25) < 0.05, \
            # 'tswap acceptance fraction != 0.25'

        assert np.all(self.sampler.acor > 0), \
            'Invalid autocorrelation lengths.'

        maxdiff = 10.0 ** logprecision

        chain = np.reshape(self.sampler.chain[0, ...],
                           (-1, self.sampler.chain.shape[-1]))

        log_volume = self.ndim * np.log(cutoff) \
            + log_unit_sphere_volume(self.ndim) \
            + 0.5 * np.log(np.linalg.det(self.cov))
        gaussian_integral = self.ndim / 2.0 * np.log(2.0 * np.pi) \
            + 0.5 * np.log(np.linalg.det(self.cov))

        logZ, dlogZ = self.sampler.thermodynamic_integration_log_evidence()

        assert np.abs(logZ - (gaussian_integral - log_volume)) < 3 * dlogZ, \
            'evidence incorrect: {0:g}+/{1:g} versus correct {2:g}'.format(logZ,
                                                                           gaussian_integral - log_volume,
                                                                           dlogZ)
        assert np.all((np.mean(chain, axis=0) - self.mean) ** 2.0 / N ** 2.0
                      < maxdiff), 'mean incorrect'
        assert np.all((np.cov(chain, rowvar=0) - self.cov) ** 2.0 / N ** 2.0
                      < maxdiff), 'covariance incorrect'

    def test_nan_logprob(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov, test_nan=True),
                               LogPriorGaussian(self.icov, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=self.Tmax)

        # If a walker is right at zero, ``logprobfn`` returns ``np.nan``.
        p0 = self.p0
        p0[-1][0][:] = 0.0

        try:
            self.check_sampler(p0=p0)
        except ValueError:
            # This should fail *immediately* with a ``ValueError``.
            pass
        else:
            assert False, 'The sampler should have failed by now.'

    def test_inf_nan_params(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov),
                               LogPriorGaussian(self.icov, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=self.Tmax)

        # Set one of the walkers to have a ``np.nan`` value.  Choose the maximum temperature as
        # we're most likely to get away with this if there's a bug.
        p0 = self.p0
        p0[-1][0][0] = np.nan

        try:
            self.check_sampler(p0=p0, cutoff=self.cutoff)
        except ValueError:
            # This should fail *immediately* with a ``ValueError``.
            pass
        else:
            assert False, 'The sampler should have failed by now.'

        # Set one of the walkers to have a ``np.inf`` value.
        p0[-1][0][0] = np.inf

        try:
            self.check_sampler(p0=p0, cutoff=self.cutoff)
        except ValueError:
            # This should fail *immediately* with a ``ValueError``.
            pass
        else:
            assert False, 'The sampler should have failed by now.'

        # Set one of the walkers to have a ``np.inf`` value.
        p0[-1][0][0] = -np.inf

        try:
            self.check_sampler(p0=p0, cutoff=self.cutoff)
        except ValueError:
            # This should fail *immediately* with a ``ValueError``.
            pass
        else:
            assert False, 'The sampler should have failed by now.'

    def test_parallel(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov),
                               LogPriorGaussian(self.icov, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=self.Tmax,
                               threads=2)
        self.check_sampler(cutoff=self.cutoff)

    def test_gaussian_adapt(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov),
                               LogPriorGaussian(self.icov, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=self.Tmax)
        self.check_sampler(cutoff=self.cutoff, adapt=True)

    def test_temp_inf(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov),
                               LogPriorGaussian(self.icov, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=np.inf)
        self.check_sampler(cutoff=self.cutoff)

    # TODO Fix this.
    # def test_blobs(self):
        # logprobfn = lambda p: (-0.5 * np.sum(p ** 2), np.random.rand())
        # self.sampler = EnsembleSampler(self.nwalkers, self.ndim, logprobfn)
        # self.check_sampler(cutoff=self.cutoff)

        # # Make sure that the shapes of everything are as expected.
        # assert (self.sampler.chain.shape == (self.nwalkers, self.N, self.ndim)
                # and len(self.sampler.blobs) == self.N
                # and len(self.sampler.blobs[0]) == self.nwalkers), \
            # 'The blob dimensions are wrong.'

        # # Make sure that the blobs aren't all the same.
        # blobs = self.sampler.blobs
        # assert np.any([blobs[-1] != blobs[i] for i in range(len(blobs) - 1)])

    def test_resume(self):
        self.sampler = s = Sampler(self.nwalkers, self.ndim,
                                   LogLikeGaussian(self.icov),
                                   LogPriorGaussian(self.icov, cutoff=self.cutoff),
                                   ntemps=self.ntemps, Tmax=self.Tmax)
        adapt = True
        N = 10

        state = s.random.get_state()
        betas = s.betas.copy()
        for i in s.sample(self.p0, iterations=N, adapt=adapt):
            pass
        assert s.chain.shape[2] == N, \
            'Expected chain of length {0}; got {1}.'.format(N, s.chain.shape[2])

        for i in s.sample(iterations=N, adapt=adapt):
            pass
        assert s.chain.shape[2] == 2 * N, \
            'Expected chain of length {0}; got {1}.'.format(2 * N, s.chain.shape[2])

        # TODO: Is this condition too strong?
        # Now do the same run afresh and compare the results.  Given the same seed, the they
        # should be identical.
        chain0 = s.chain.copy()
        betas0 = s.betas.copy()
        s.reset(betas=betas)
        s.random.set_state(state)
        for i in s.sample(self.p0, iterations=2 * N, adapt=adapt):
            pass
        assert np.all(s.chain == chain0), 'Chains don\'t match!'
        assert np.all(s.betas == betas0), 'Ladders don\'t match!'
