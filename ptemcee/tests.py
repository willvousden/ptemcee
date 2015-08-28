#!/usr/bin/env python
# encoding: utf-8
'''
Defines various nose unit tests.

'''

from __future__ import division

import numpy as np
from .sampler import Sampler

logprecision = -4

def logprob_gaussian(x, icov):
    return -np.dot(x, np.dot(icov, x)) / 2.0

def logprob_gaussian_nan(x, icov):
    # If any of walker's parameters are zeros, return NaN.
    if not (np.array(x)).any():
        return np.nan
    else:
        return logprob_gaussian(x, icov)

def logprob_gaussian_inf(x, icov):
    # If any of walker's parameters are negative, return -inf.
    if (np.array(x) < 0).any():
        return -np.inf
    else:
        return logprob_gaussian(x, icov)

def log_unit_sphere_volume(ndim):
    if ndim % 2 == 0:
        logfactorial = 0.0
        for i in range(1, ndim // 2 + 1):
            logfactorial += np.log(i)
        return ndim / 2.0 * np.log(np.pi) - logfactorial
    else:
        logfactorial = 0.0
        for i in range(1, ndim + 1, 2):
            logfactorial += np.log(i)
        return (ndim + 1) / 2.0 * np.log(2.0) \
            + (ndim - 1) / 2.0 * np.log(np.pi) - logfactorial

class LogLikeGaussian(object):
    def __init__(self, icov, test_nan=False, test_inf=False):
        '''Initialize a gaussian PDF with the given inverse covariance
        matrix.  If not ``None``, ``cutoff`` truncates the PDF at the
        given number of sigma from the origin (i.e. the PDF is
        non-zero only on an ellipse aligned with the principal axes of
        the distribution).  Without this cutoff, thermodynamic
        integration with a flat prior is logarithmically divergent.'''

        self.icov = icov
        self.test_nan = test_nan
        self.test_inf = test_inf

    def __call__(self, x):
        if self.test_nan:
            f = logprob_gaussian_nan
        elif self.test_inf:
            f = logprob_gaussian_inf
        else:
            f = logprob_gaussian

        return f(x, self.icov)

class LogPriorGaussian(object):
    def __init__(self, icov, cutoff=None):
        self.icov = icov
        self.cutoff = cutoff

    def __call__(self, x):
        contour = logprob_gaussian(x, self.icov)

        if self.cutoff is not None:
            if -contour > self.cutoff * self.cutoff / 2.0:
                return -np.inf
            else:
                return 0.0
        else:
            return 0.0

class Tests(object):
    def setUp(self):
        np.seterr(all='raise')

        self.nwalkers = 100
        self.ndim = 5

        self.ntemps = 10
        self.Tmax = 250
        self.cutoff = 10

        self.N = 1000

        self.mean = np.zeros(self.ndim)
        self.sqrtcov = 0.5 - np.random.rand(self.ndim, self.ndim)
        self.sqrtcov = np.triu(self.sqrtcov)
        self.sqrtcov += self.sqrtcov.T - np.diag(self.sqrtcov.diagonal())
        self.cov = np.dot(self.sqrtcov, self.sqrtcov)
        self.icov = np.linalg.inv(self.cov)
        self.icov_unit = np.eye(self.ndim)

        # Draw samples from unit ball.
        nsamples = self.ntemps * self.nwalkers
        x = np.random.randn(nsamples, self.ndim)
        x /= np.linalg.norm(x, axis=-1).reshape((nsamples, 1))
        x *= np.random.rand(nsamples).reshape((nsamples, 1)) ** (1 / self.ndim)

        # Now transform them to cover the prior volume.
        self.p0_unit = x * self.cutoff
        self.p0 = np.dot(x, self.sqrtcov)

        self.p0_unit = self.p0_unit.reshape(self.ntemps, self.nwalkers, self.ndim)
        self.p0 = self.p0.reshape(self.ntemps, self.nwalkers, self.ndim)

    def check_sampler(self,
                      cutoff=None, N=None, p0=None,
                      adapt=False, weak=False, fail=False):
        if cutoff is None:
            cutoff = self.cutoff
        if N is None:
            N = self.N
        if p0 is None:
            p0 = self.p0

        if fail:
            # Require the sampler to fail before it even starts.
            try:
                for x in self.sampler.sample(p0, iterations=N, adapt=adapt):
                    assert False, \
                        'Sampler should have failed by now.'
            except Exception as e:
                # If a type was specified, require that the sampler fail with this exception type.
                assert type(fail) is not type or type(e) is fail, \
                    'Sampler failed with unexpected exception type.'
                return
        else:
            for p, logpost, loglike, ratios in self.sampler.sample(p0, iterations=N, adapt=adapt, swap_ratios=True):
                assert np.all(logpost > -np.inf) and np.all(loglike > -np.inf), \
                    'Invalid posterior/likelihood values; outside posterior support.'
                assert np.all(ratios >= 0) and np.all(ratios <= 1), \
                    'Invalid swap ratios.'
                assert logpost.shape == loglike.shape == p.shape[:-1], \
                    'Sampler output shapes invalid.'
                assert p.shape[-1] == self.ndim, \
                    'Sampler output shapes invalid.'
                assert ratios.shape[0] == logpost.shape[0] - 1 and len(ratios.shape) == 1, \
                    'Sampler output shapes invalid.'
                assert np.all(self.sampler.betas >= 0), \
                    'Negative temperatures!'
                assert np.all(np.diff(self.sampler.betas) != 0), \
                    'Temperatures have coalesced.'
                assert np.all(np.diff(self.sampler.betas) < 0), \
                    'Temperatures incorrectly ordered.'

        assert np.all(self.sampler.acor > 0), \
            'Invalid autocorrelation lengths.'

        if not weak:
            # Weaker assertions on acceptance fraction
            assert np.mean(self.sampler.acceptance_fraction) > 0.1, \
                'acceptance fraction < 0.1'
            assert np.mean(self.sampler.tswap_acceptance_fraction) > 0.1, \
                'tswap acceptance fraction < 0.1'
            # TODO
            # assert abs(self.sampler.tswap_acceptance_fraction[0] - 0.25) < 0.05, \
                # 'tswap acceptance fraction != 0.25'

            chain = np.reshape(self.sampler.chain[0, ...],
                               (-1, self.sampler.chain.shape[-1]))

            log_volume = self.ndim * np.log(cutoff) \
                + log_unit_sphere_volume(self.ndim) \
                + 0.5 * np.log(np.linalg.det(self.cov))
            gaussian_integral = self.ndim / 2.0 * np.log(2.0 * np.pi) \
                + 0.5 * np.log(np.linalg.det(self.cov))

            logZ, dlogZ = self.sampler.log_evidence_estimate()

            assert np.abs(logZ - (gaussian_integral - log_volume)) < 3 * dlogZ, \
                'evidence incorrect: {0:g}+/{1:g} versus correct {2:g}'.format(logZ,
                                                                               gaussian_integral - log_volume,
                                                                               dlogZ)
            maxdiff = 10.0 ** logprecision
            assert np.all((np.mean(chain, axis=0) - self.mean) ** 2.0 / N ** 2.0
                          < maxdiff), 'mean incorrect'
            assert np.all((np.cov(chain, rowvar=0) - self.cov) ** 2.0 / N ** 2.0
                          < maxdiff), 'covariance incorrect'

    def test_prior_support(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov_unit),
                               LogPriorGaussian(self.icov_unit, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=self.Tmax)

        # What happens when we start the sampler outside our prior support?
        self.p0_unit[0][0][0] = 1e6 * self.cutoff
        self.check_sampler(p0=self.p0_unit, fail=ValueError)

    def test_likelihood_support(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov_unit, test_inf=True),
                               LogPriorGaussian(self.icov_unit, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=self.Tmax)

        # What happens when we start the sampler outside our likelihood support?  Give some walkers a
        # negative parameter value, where the likelihood is unsupported.
        self.p0_unit[0][0][0] = -1
        self.check_sampler(p0=self.p0_unit, fail=ValueError)

    def test_nan_logprob(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov_unit, test_nan=True),
                               LogPriorGaussian(self.icov_unit, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=self.Tmax)

        # If a walker is right at zero, ``logprobfn`` returns ``np.nan``; sampler should fail with a
        # ``ValueError``.
        self.p0_unit[-1][0][:] = 0
        self.check_sampler(p0=self.p0_unit, fail=ValueError)

    def test_inf_logprob(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov_unit, test_inf=True),
                               LogPriorGaussian(self.icov_unit, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=np.inf)

        # If a walker has any parameter negative, ``logprobfn`` returns ``-np.inf``.  Start the
        # ensembles in the all-positive part of the parameter space, then run for long enough for
        # sampler to migrate into negative parts.  (We can't start outside the posterior support, or
        # the sampler will fail).  The sampler should be happy with this; otherwise, a
        # FloatingPointError will be thrown by Numpy.  Don't bother checking the results because
        # this posterior is difficult to sample.
        self.check_sampler(p0=np.abs(self.p0_unit), weak=True)

    def test_inf_nan_params(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov_unit),
                               LogPriorGaussian(self.icov_unit, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=self.Tmax)

        # Set one of the walkers to have a ``np.nan`` value.  Choose the maximum temperature as
        # we're most likely to get away with this if there's a bug.
        self.p0_unit[-1][0][0] = np.nan
        self.check_sampler(p0=self.p0_unit, fail=ValueError)

        # Set one of the walkers to have a ``np.inf`` value.
        self.p0_unit[-1][0][0] = np.inf
        self.check_sampler(p0=self.p0_unit, fail=ValueError)

        # Set one of the walkers to have a ``-np.inf`` value.
        self.p0_unit[-1][0][0] = -np.inf
        self.check_sampler(p0=self.p0_unit, fail=ValueError)

    def test_parallel(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov),
                               LogPriorGaussian(self.icov, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=self.Tmax,
                               threads=2)
        self.check_sampler()

    def test_temp_inf(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov),
                               LogPriorGaussian(self.icov, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=np.inf)
        self.check_sampler()

    def test_gaussian_adapt(self):
        self.sampler = Sampler(self.nwalkers, self.ndim,
                               LogLikeGaussian(self.icov),
                               LogPriorGaussian(self.icov, cutoff=self.cutoff),
                               ntemps=self.ntemps, Tmax=self.Tmax)
        self.check_sampler(adapt=True)

    def test_run_mcmc(self):
        """
        Check that ``Sampler.run_mcmc()`` is equivalent to ``Sampler.sample()``.

        """

        N = 10
        self.sampler = s = Sampler(self.nwalkers, self.ndim,
                                   LogLikeGaussian(self.icov),
                                   LogPriorGaussian(self.icov, cutoff=self.cutoff),
                                   ntemps=self.ntemps, Tmax=self.Tmax)

        state = s.random.get_state()
        betas = s.betas.copy()
        s.run_mcmc(self.p0, iterations=N, adapt=True)

        chain0 = s.chain.copy()
        betas0 = s.betas.copy()
        s.reset(betas=betas)
        s.random.set_state(state)
        for x in s.run_mcmc(self.p0, iterations=N, adapt=True):
            pass
        assert np.all(s.chain == chain0), \
            'Chains don\'t match!'
        assert np.all(s.betas == betas0), \
            'Ladders don\'t match!'

    def test_resume(self):
        N = 10
        self.sampler = s = Sampler(self.nwalkers, self.ndim,
                                   LogLikeGaussian(self.icov),
                                   LogPriorGaussian(self.icov, cutoff=self.cutoff),
                                   ntemps=self.ntemps, Tmax=self.Tmax)

        state = s.random.get_state()
        betas = s.betas.copy()
        s.run_mcmc(self.p0, iterations=N, adapt=True)
        assert s.chain.shape[2] == N, \
            'Expected chain of length {0}; got {1}.'.format(N, s.chain.shape[2])

        s.run_mcmc(iterations=N, adapt=True)
        assert s.chain.shape[2] == 2 * N, \
            'Expected chain of length {0}; got {1}.'.format(2 * N, s.chain.shape[2])

        # TODO: Is this condition too strong?
        # Now do the same run afresh and compare the results.  Given the same seed, the they
        # should be identical.
        chain0 = s.chain.copy()
        betas0 = s.betas.copy()
        s.reset(betas=betas)
        s.random.set_state(state)
        s.run_mcmc(self.p0, iterations=2 * N, adapt=True)
        assert np.all(s.chain == chain0), \
            'Chains don\'t match!'
        assert np.all(s.betas == betas0), \
            'Ladders don\'t match!'
