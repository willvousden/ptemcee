#!/usr/bin/env python
# encoding: utf-8
"""
Defines various pytest unit tests.

"""

from __future__ import absolute_import, print_function, division

import itertools
import numpy as np

from numpy.random.mtrand import RandomState

from .sampler import Sampler, make_ladder
from .interruptible_pool import Pool

logprecision = -4


def logprob_gaussian(x, icov):
    return -np.dot(x, np.dot(icov, x)) / 2


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
        logfactorial = 0
        for i in range(1, ndim // 2 + 1):
            logfactorial += np.log(i)
        return ndim / 2 * np.log(np.pi) - logfactorial
    else:
        logfactorial = 0
        for i in range(1, ndim + 1, 2):
            logfactorial += np.log(i)
        return (ndim + 1) / 2 * np.log(2) \
            + (ndim - 1) / 2 * np.log(np.pi) - logfactorial


class LogLikeGaussian(object):
    def __init__(self, icov, test_nan=False, test_inf=False):
        """
        Initialize a gaussian PDF with the given inverse covariance matrix.  If
        not ``None``, ``cutoff`` truncates the PDF at the given number of sigma
        from the origin (i.e., the PDF is non-zero only on an ellipse aligned
        with the principal axes of the distribution).  Without this cutoff,
        thermodynamic integration with a flat prior is logarithmically
        divergent.

        """

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
            if -contour > self.cutoff * self.cutoff / 2:
                return -np.inf
            else:
                return 0
        else:
            return 0


class Tests(object):
    sampler = None  # type: Sampler

    @classmethod
    def setup_class(cls):
        np.seterr(all='raise')

        cls.nwalkers = 100  # type: int
        cls.ndim = 5  # type: int

        cls.ntemps = 10  # type: int
        cls.Tmax = 250  # type: float
        cls.cutoff = 10  # type: float

        cls.N = 1000  # type: int

        cls.mean = np.zeros(cls.ndim)  # type: np.ndarray
        sqrtcov = 0.5 - np.random.rand(cls.ndim, cls.ndim)
        sqrtcov = np.triu(sqrtcov)
        sqrtcov += sqrtcov.T - np.diag(sqrtcov.diagonal())
        cls.cov = np.dot(sqrtcov, sqrtcov)  # type: np.ndarray
        cls.icov = np.linalg.inv(cls.cov)  # type: np.ndarray
        cls.icov_unit = np.eye(cls.ndim)  # type: np.ndarray

        # Draw samples from unit ball.
        nsamples = cls.ntemps * cls.nwalkers
        x = np.random.randn(nsamples, cls.ndim)
        x /= np.linalg.norm(x, axis=-1).reshape((nsamples, 1))
        x *= np.random.rand(nsamples).reshape((nsamples, 1)) ** (1 / cls.ndim)

        # Now transform them to cover the prior volume.
        cls.p0_unit = x * cls.cutoff  # type: np.ndarray
        cls.p0 = np.dot(x, sqrtcov)  # type: np.ndarray

        cls.p0_unit = cls.p0_unit.reshape(cls.ntemps, cls.nwalkers, cls.ndim)  # type: np.ndarray
        cls.p0 = cls.p0.reshape(cls.ntemps, cls.nwalkers, cls.ndim)  # type: np.ndarray

    def check_sampler(self, sampler, p0, weak=False, fail=False):
        """
        Check that the sampler is behaving itself.

        Parameters
        ----------
        sampler : Sampler
            The sampler to check.
        p0 : float, optional
            The initial positions at which to start the sampler's walkers.
        weak : bool, optional
            If ``True``, just check that the sampler ran without errors; don't
            check any of the results.
        fail : :class:`Exception`, optional
            If specified, assert that the sampler fails with the given
            exception type.

        """

        if fail:
            # Require the sampler to fail before it even starts.
            try:
                chain = sampler.chain(p0)
                for x in chain.iterate(self.N):
                    assert False, \
                        'Sampler should have failed by now.'
            except Exception as e:
                # If a type was specified, require that the sampler fail with this exception type.
                if type(e) is fail:
                    return
                else:
                    raise
        else:
            chain = sampler.chain(p0)
            for ensemble in chain.iterate(self.N):
                ratios = ensemble.swaps_accepted / ensemble.swaps_proposed

                assert np.all(ensemble.logP > -np.inf) and np.all(ensemble.logP > -np.inf), \
                    'Invalid posterior/likelihood values; outside posterior support.'
                assert np.all(ratios >= 0) and np.all(ratios <= 1), \
                    'Invalid swap ratios.'
                assert ensemble.logP.shape == ensemble.logP.shape == ensemble.x.shape[:-1], \
                    'Sampler output shapes invalid.'
                assert ensemble.x.shape[-1] == self.ndim, \
                    'Sampler output shapes invalid.'
                assert ratios.shape[0] == ensemble.logP.shape[0] - 1 and len(ratios.shape) == 1, \
                    'Sampler output shapes invalid.'
                assert np.all(ensemble.betas >= 0), \
                    'Negative temperatures!'
                assert np.all(np.diff(ensemble.betas) != 0), \
                    'Temperatures have coalesced.'
                assert np.all(np.diff(ensemble.betas) < 0), \
                    'Temperatures incorrectly ordered.'

        assert np.all(chain.get_acts() > 0), \
            'Invalid autocorrelation lengths.'

        if not weak:
            # Weaker assertions on acceptance fraction
            assert np.mean(chain.jump_acceptance_ratio) > 0.1, \
                'Acceptance fraction < 0.1'
            assert np.mean(chain.swap_acceptance_ratio) > 0.1, \
                'Temperature swap acceptance fraction < 0.1.'

            # TODO: Why doesn't this work?
            # if sampler.adaptive:
            #     assert abs(chain.swap_acceptance_ratio[0] - 0.25) < 0.05, \
            #         'Swap acceptance ratio != 0.25'

            data = np.reshape(chain.x[0, ...], (-1, chain.x.shape[-1]))

            log_volume = self.ndim * np.log(self.cutoff) \
                + log_unit_sphere_volume(self.ndim) \
                + 0.5 * np.log(np.linalg.det(self.cov))
            gaussian_integral = self.ndim / 2 * np.log(2 * np.pi) \
                + 0.5 * np.log(np.linalg.det(self.cov))

            logZ, dlogZ = chain.log_evidence_estimate()

            assert np.abs(logZ - (gaussian_integral - log_volume)) < 3 * dlogZ, \
                'Evidence incorrect: {:g}+/{:g} versus correct {:g}.' \
                .format(logZ, gaussian_integral - log_volume, dlogZ)
            maxdiff = 10 ** logprecision
            assert np.all((np.mean(data, axis=0) - self.mean) ** 2 / self.N ** 2
                          < maxdiff), 'Mean incorrect.'
            assert np.all((np.cov(data, rowvar=False) - self.cov) ** 2 / self.N ** 2
                          < maxdiff), 'Covariance incorrect.'

    def test_prior_support(self):
        sampler = Sampler(self.nwalkers, self.ndim,
                          LogLikeGaussian(self.icov_unit),
                          LogPriorGaussian(self.icov_unit, cutoff=self.cutoff),
                          betas=make_ladder(self.ndim, self.ntemps, Tmax=self.Tmax))

        # What happens when we start the sampler outside our prior support?
        self.p0_unit[0][0][0] = 1e6 * self.cutoff
        self.check_sampler(sampler, p0=self.p0_unit, fail=ValueError)

    def test_likelihood_support(self):
        sampler = Sampler(self.nwalkers, self.ndim,
                          LogLikeGaussian(self.icov_unit, test_inf=True),
                          LogPriorGaussian(self.icov_unit, cutoff=self.cutoff),
                          betas=make_ladder(self.ndim, self.ntemps, Tmax=self.Tmax))

        # What happens when we start the sampler outside our likelihood
        # support?  Give some walkers a negative parameter value, where the
        # likelihood is unsupported.
        self.p0_unit[0][0][0] = -1
        self.check_sampler(sampler, p0=self.p0_unit, fail=ValueError)

    def test_nan_logprob(self):
        sampler = Sampler(self.nwalkers, self.ndim,
                          LogLikeGaussian(self.icov_unit, test_nan=True),
                          LogPriorGaussian(self.icov_unit, cutoff=self.cutoff),
                          betas=make_ladder(self.ndim, self.ntemps, Tmax=self.Tmax))

        # If a walker is right at zero, ``logprobfn`` returns ``np.nan``;
        # sampler should fail with a ``ValueError``.
        self.p0_unit[-1][0][:] = 0
        self.check_sampler(sampler, p0=self.p0_unit, fail=ValueError)

    def test_inf_logprob(self):
        """
        If a walker has any parameter negative, ``logprobfn`` returns
        ``-np.inf``.  Start the ensembles in the all-positive part of the
        parameter space, then run for long enough for sampler to migrate into
        negative parts.  (We can't start outside the posterior support, or the
        sampler will fail).  The sampler should be happy with this; otherwise,
        a FloatingPointError will be thrown by Numpy.  Don't bother checking
        the results because this posterior is difficult to sample.

        """
        sampler = Sampler(self.nwalkers, self.ndim,
                          LogLikeGaussian(self.icov_unit, test_inf=True),
                          LogPriorGaussian(self.icov_unit, cutoff=self.cutoff),
                          betas=make_ladder(self.ndim, self.ntemps, np.inf))

        self.check_sampler(sampler, p0=np.abs(self.p0_unit), weak=True)

    def test_inf_nan_params(self):
        sampler = Sampler(self.nwalkers, self.ndim,
                          LogLikeGaussian(self.icov_unit),
                          LogPriorGaussian(self.icov_unit, cutoff=self.cutoff),
                          betas=make_ladder(self.ndim, self.ntemps, Tmax=self.Tmax))

        # Set one of the walkers to have a ``np.nan`` value.  Choose the
        # maximum temperature as we're most likely to get away with this if
        # there's a bug.
        self.p0_unit[-1][0][0] = np.nan
        self.check_sampler(sampler, p0=self.p0_unit, fail=ValueError)

        # Set one of the walkers to have a ``np.inf`` value.
        self.p0_unit[-1][0][0] = np.inf
        self.check_sampler(sampler, p0=self.p0_unit, fail=ValueError)

        # Set one of the walkers to have a ``-np.inf`` value.
        self.p0_unit[-1][0][0] = -np.inf
        self.check_sampler(sampler, p0=self.p0_unit, fail=ValueError)

    def test_parallel(self):
        sampler = Sampler(self.nwalkers, self.ndim,
                          LogLikeGaussian(self.icov),
                          LogPriorGaussian(self.icov, cutoff=self.cutoff),
                          betas=make_ladder(self.ndim, self.ntemps, Tmax=self.Tmax),
                          mapper=Pool(2).map)
        self.check_sampler(sampler, p0=self.p0)

    def test_temp_inf(self):
        sampler = Sampler(self.nwalkers, self.ndim,
                          LogLikeGaussian(self.icov),
                          LogPriorGaussian(self.icov, cutoff=self.cutoff),
                          betas=make_ladder(self.ndim, self.ntemps, Tmax=self.Tmax))
        self.check_sampler(sampler, p0=self.p0)

    def test_betas(self):
        sampler = Sampler(self.nwalkers, self.ndim,
                          LogLikeGaussian(self.icov),
                          LogPriorGaussian(self.icov, cutoff=self.cutoff),
                          betas=20)
        assert sampler.betas.shape[0] == 20

    def test_gaussian_adapt(self):
        sampler = Sampler(self.nwalkers, self.ndim,
                          LogLikeGaussian(self.icov),
                          LogPriorGaussian(self.icov, cutoff=self.cutoff),
                          adaptive=True,
                          betas=make_ladder(self.ndim, self.ntemps, Tmax=self.Tmax))
        self.check_sampler(sampler, p0=self.p0)

    def test_ensemble(self):
        """
        Test that various ways of running the sampler are equivalent.

        """

        sampler = Sampler(self.nwalkers, self.ndim,
                          LogLikeGaussian(self.icov),
                          LogPriorGaussian(self.icov, cutoff=self.cutoff),
                          adaptive=True,
                          betas=make_ladder(self.ndim, self.ntemps, Tmax=self.Tmax))

        N = 10
        thin_by = 2
        seed = 1
        ensemble = sampler.ensemble(self.p0, RandomState(seed))
        chain = sampler.chain(self.p0, RandomState(seed), thin_by)
        samples = sampler.sample(self.p0, RandomState(seed), thin_by)

        betas1 = np.empty((N, self.ntemps))
        x1 = np.empty((N, self.ntemps, self.nwalkers, self.ndim))
        for i in range(N):
            for _ in range(thin_by):
                ensemble.step()
            x1[i] = ensemble.x
            betas1[i] = ensemble.betas

        betas2 = np.empty((N, self.ntemps))
        x2 = np.empty((N, self.ntemps, self.nwalkers, self.ndim))
        for i, e in enumerate(itertools.islice(samples, N)):
            x2[i] = e.x
            betas2[i] = e.betas

        jr, sr = chain.run(N)
        assert (chain.jump_acceptance_ratio == jr).all(), 'Jump acceptance ratios don\'t match.'
        assert (chain.swap_acceptance_ratio == sr).all(), 'Swap acceptance ratios don\'t match.'
        assert (chain.x == x1).all(), 'Chains don\'t match.'
        assert (chain.x == x2).all(), 'Chains don\'t match.'
        assert (chain.betas == betas1).all(), 'Ladders don\'t match.'
        assert (chain.betas == betas2).all(), 'Ladders don\'t match.'

    def test_resume(self):
        sampler = Sampler(self.nwalkers, self.ndim,
                          LogLikeGaussian(self.icov),
                          LogPriorGaussian(self.icov, cutoff=self.cutoff),
                          adaptive=True,
                          betas=make_ladder(self.ndim, self.ntemps, Tmax=self.Tmax))

        N = 10
        thin_by = 2
        seed = 1

        # Run the chain in two parts.
        chain1 = sampler.chain(self.p0, RandomState(seed), thin_by)
        jr, sr = chain1.run(N)
        assert (0 <= jr).all() and (jr <= 1).all()
        assert (0 <= sr).all() and (sr <= 1).all()
        assert chain1.x.shape[0] == N

        jr, sr = chain1.run(N)
        assert (0 <= jr).all() and (jr <= 1).all()
        assert (0 <= sr).all() and (sr <= 1).all()
        assert chain1.x.shape[0] == 2 * N

        # Now do the same run afresh and compare the results.  Given the same seed, the they should be identical.
        chain2 = sampler.chain(self.p0, RandomState(seed), thin_by)
        jr, sr = chain2.run(2 * N)
        assert (0 <= jr).all() and (jr <= 1).all()
        assert (0 <= sr).all() and (sr <= 1).all()
        assert chain2.x.shape[0] == 2 * N

        assert (chain1.x == chain2.x).all(), 'Chains don\'t match.'
        assert (chain1.betas == chain2.betas).all(), 'Ladders don\'t match.'
