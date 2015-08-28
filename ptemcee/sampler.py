#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Sampler", "default_beta_ladder"]

import numpy as np
import multiprocessing as multi
from . import util

def default_beta_ladder(ndim, ntemps=None, Tmax=None):
    """
    Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
    arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:

    Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
    this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
    <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
    ``ntemps`` is also specified.

    :param ndim:
        The number of dimensions in the parameter space.

    :param ntemps: (optional)
        If set, the number of temperatures to generate.

    :param Tmax: (optional)
        If set, the maximum temperature for the ladder.

    Temperatures are chosen according to the following algorithm:

    * If neither ``ntemps`` nor ``Tmax`` is specified, raise an exception (insufficient
      information).
    * If ``ntemps`` is specified but not ``Tmax``, return a ladder spaced so that a Gaussian
      posterior would have a 25% temperature swap acceptance ratio.
    * If ``Tmax`` is specified but not ``ntemps``:

      * If ``Tmax = inf``, raise an exception (insufficient information).
      * Else, space chains geometrically as above (for 25% acceptance) until ``Tmax`` is reached.

    * If ``Tmax`` and ``ntemps`` are specified:

      * If ``Tmax = inf``, place one chain at ``inf`` and ``ntemps-1`` in a 25% geometric spacing.
      * Else, use the unique geometric spacing defined by ``ntemps`` and ``Tmax``.

    """

    if type(ndim) != int or ndim < 1:
        raise ValueError('Invalid number of dimensions specified.')
    if ntemps is None and Tmax is None:
        raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
    if Tmax is not None and Tmax <= 1:
        raise ValueError('``Tmax`` must be greater than 1.')
    if ntemps is not None and (type(ntemps) != int or ntemps < 1):
        raise ValueError('Invalid number of temperatures specified.')

    tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                      2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                      2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                      1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                      1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                      1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                      1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                      1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                      1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                      1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                      1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                      1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                      1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                      1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                      1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                      1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                      1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                      1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                      1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                      1.26579, 1.26424, 1.26271, 1.26121,
                      1.25973])

    if ndim > tstep.shape[0]:
        # An approximation to the temperature step at large
        # dimension
        tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
    else:
        tstep = tstep[ndim-1]

    appendInf = False
    if Tmax == np.inf:
        appendInf = True
        Tmax = None
        ntemps = ntemps - 1

    if ntemps is not None:
        if Tmax is None:
            # Determine Tmax from ntemps.
            Tmax = tstep ** (ntemps - 1)
    else:
        if Tmax is None:
            raise ValueError('Must specify at least one of ``ntemps'' and finite ``Tmax``.')

        # Determine ntemps from Tmax.
        ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

    betas = np.logspace(0, -np.log10(Tmax), ntemps)
    if appendInf:
        # Use a geometric spacing, but replace the top-most temperature with infinity.
        betas = np.concatenate((betas, [0]))

    return betas

class LikePriorEvaluator(object):
    """
    Wrapper class for logl and logp.

    """

    def __init__(self, logl, logp,
                 loglargs=[], logpargs=[],
                 loglkwargs={}, logpkwargs={}):
        self.logl = logl
        self.logp = logp
        self.loglargs = loglargs
        self.logpargs = logpargs
        self.loglkwargs = loglkwargs
        self.logpkwargs = logpkwargs

    def __call__(self, x):
        lp = self.logp(x, *self.logpargs, **self.logpkwargs)
        if np.isnan(lp):
            raise ValueError('Prior function returned NaN.')

        if lp == float('-inf'):
            # Can't return -inf, since this messes with beta=0 behaviour.
            ll = 0
        else:
            ll = self.logl(x, *self.loglargs, **self.loglkwargs)
            if np.isnan(ll).any():
                raise ValueError('Log likelihood function returned NaN.')

        return ll, lp

class Sampler(object):
    """
    A parallel-tempered ensemble sampler, using :class:`EnsembleSampler`
    for sampling within each parallel chain.

    :param nwalkers:
        The number of ensemble walkers at each temperature.

    :param dim:
        The dimension of parameter space.

    :param betas: (optional)
        Array giving the inverse temperatures, :math:`\\beta=1/T`, used in the ladder.  The default
        is chosen according to :function:`default_beta_ladder` using ``ntemps`` and ``Tmax``.

    :param ntemps: (optional)
        If set, the number of temperatures to use.

    :param Tmax: (optional)
        If set, the maximum temperature for the ladder.

    :param logl:
        The log-likelihood function.

    :param logp:
        The log-prior function.

    :param threads: (optional)
        The number of parallel threads to use in sampling.

    :param pool: (optional)
        Alternative to ``threads``.  Any object that implements a
        ``map`` method compatible with the built-in ``map`` will do
        here.  For example, :class:`multi.Pool` will do.

    :param a: (optional)
        Proposal scale factor.

    :param loglargs: (optional)
        Positional arguments for the log-likelihood function.

    :param logpargs: (optional)
        Positional arguments for the log-prior function.

    :param loglkwargs: (optional)
        Keyword arguments for the log-likelihood function.

    :param logpkwargs: (optional)
        Keyword arguments for the log-prior function.

    :param adaptation_lag: (optional)
        Time lag for temperature dynamics decay. Default: 10000.

    :param adaptation_time: (optional)
        Time-scale for temperature dynamics.  Default: 100.

    """
    def __init__(self, nwalkers, dim, logl, logp,
                 ntemps=None, Tmax=None, betas=None,
                 threads=1, pool=None, a=2.0,
                 loglargs=[], logpargs=[],
                 loglkwargs={}, logpkwargs={},
                 adaptation_lag=10000, adaptation_time=100,
                 random=None):
        if random is None:
            self._random = np.random.mtrand.RandomState()
        else:
            self._random = random

        self._likeprior = LikePriorEvaluator(logl, logp, loglargs, logpargs, loglkwargs, logpkwargs)
        self.a = a
        self.nwalkers = nwalkers
        self.dim = dim
        self.adaptation_time = adaptation_time
        self.adaptation_lag = adaptation_lag

        # Set temperature ladder.  Append beta=0 to generated ladder.
        if betas is not None:
            self._betas = np.array(betas).copy()
        else:
            self._betas = default_beta_ladder(self.dim, ntemps=ntemps, Tmax=Tmax)

        # Make sure ladder is ascending in temperature.
        self._betas[::-1].sort()

        if self.nwalkers % 2 != 0:
            raise ValueError('The number of walkers must be even.')
        if self.nwalkers < 2 * self.dim:
            raise ValueError('The number of walkers must be greater than ``2*dimension``.')

        self.pool = pool
        if threads > 1 and pool is None:
            self.pool = multi.Pool(threads)

        self.reset()

    def reset(self, random=None, betas=None, time=None):
        """
        Clear the ``time``, ``chain``, ``logposterior``,
        ``loglikelihood``,  ``acceptance_fraction``,
        ``tswap_acceptance_fraction`` stored properties.

        """

        # Reset chain.
        self._chain = None
        self._logposterior = None
        self._loglikelihood = None
        self._beta_history = None

        # Reset sampler state.
        self._time = 0
        self._p0 = None
        self._logposterior0 = None
        self._loglikelihood0 = None
        if betas is not None:
            self._betas = betas

        self.nswap = np.zeros(self.ntemps, dtype=np.float)
        self.nswap_accepted = np.zeros(self.ntemps, dtype=np.float)

        self.nprop = np.zeros((self.ntemps, self.nwalkers), dtype=np.float)
        self.nprop_accepted = np.zeros((self.ntemps, self.nwalkers), dtype=np.float)

        if random is not None:
            self._random = random
        if time is not None:
            self._time = time

    def run_mcmc(self, *args, **kwargs):
        """
        Identical to ``sample``, but returns the final ensemble and discards intermediate ensembles.

        """
        for x in self.sample(*args, **kwargs):
            pass
        return x

    def sample(self, p0=None,
               iterations=1, thin=1,
               storechain=True, adapt=False,
               swap_ratios=False):
        """
        Advance the chains ``iterations`` steps as a generator.

        :param p0:
            The initial positions of the walkers.  Shape should be
            ``(ntemps, nwalkers, dim)``.  Can be omitted if resuming
            from a previous run.

        :param iterations: (optional)
            The number of iterations to perform.

        :param thin: (optional)
            The number of iterations to perform between saving the
            state to the internal chain.

        :param storechain: (optional)
            If ``True`` store the iterations in the ``chain``
            property.

        :param adapt: (optional)
            If ``True``, the temperature ladder is dynamically adapted as the sampler runs to
            achieve uniform swap acceptance ratios between adjacent chains.  See `arXiv:1501.05823
            <http://arxiv.org/abs/1501.05823>`_ for details.

        :param swap_ratios: (optional)
            If ``True``, also yield the instantaneous inter-chain acceptance ratios in the fourth
            element of the yielded tuple.

        At each iteration, this generator yields

        * ``p``, the current position of the walkers.

        * ``logpost``, the current posterior values for the walkers.

        * ``loglike``, the current likelihood values for the walkers.

        * ``ratios``, the instantaneous inter-chain acceptance ratios (if requested).

        """

        # Set initial walker positions.
        if p0 is not None:
            # Start anew.
            self._p0 = p = np.array(p0).copy()
            self._logposterior0 = None
            self._loglikelihood0 = None
        elif self._p0 is not None:
            # Now, where were we?
            p = self._p0
        else:
            raise ValueError('Initial walker positions not specified.')

        # Check for dodgy inputs.
        if np.any(np.isinf(p)):
            raise ValueError('At least one parameter value was infinite.')
        if np.any(np.isnan(p)):
            raise ValueError('At least one parameter value was NaN.')

        # If we have no likelihood or prior values, compute them.
        if self._logposterior0 is None or self._loglikelihood0 is None:
            logl, logp = self._evaluate(p)
            logpost = self._tempered_likelihood(logl) + logp

            self._loglikelihood0 = logl
            self._logposterior0 = logpost
        else:
            logl = self._loglikelihood0
            logpost = self._logposterior0

        if (logpost == -np.inf).any():
            raise ValueError('Attempting to start with samples outside posterior support.')

        # Expand the chain in advance of the iterations
        if storechain:
            isave = self._expand_chain(iterations // thin)

        for i in range(iterations):
            for j in [0, 1]:
                # Get positions of walkers to be updated and walker to be sampled.
                jupdate = j
                jsample = (j + 1) % 2
                pupdate = p[:, jupdate::2, :]
                psample = p[:, jsample::2, :]

                zs = np.exp(self._random.uniform(low=-np.log(self.a),
                                                 high=np.log(self.a),
                                                 size=(self.ntemps, self.nwalkers//2)))

                qs = np.zeros((self.ntemps, self.nwalkers//2, self.dim))
                for k in range(self.ntemps):
                    js = self._random.randint(0, high=self.nwalkers // 2,
                                              size=self.nwalkers // 2)
                    qs[k, :, :] = psample[k, js, :] + zs[k, :].reshape(
                        (self.nwalkers // 2, 1)) * (pupdate[k, :, :] -
                                                   psample[k, js, :])

                qslogl, qslogp = self._evaluate(qs)
                qslogpost = self._tempered_likelihood(qslogl) + qslogp

                logpaccept = self.dim*np.log(zs) + qslogpost \
                    - logpost[:, jupdate::2]
                logr = np.log(self._random.uniform(low=0.0, high=1.0,
                                                   size=(self.ntemps,
                                                         self.nwalkers//2)))

                accepts = logr < logpaccept
                accepts = accepts.flatten()

                pupdate.reshape((-1, self.dim))[accepts, :] = \
                    qs.reshape((-1, self.dim))[accepts, :]
                logpost[:, jupdate::2].reshape((-1,))[accepts] = \
                    qslogpost.reshape((-1,))[accepts]
                logl[:, jupdate::2].reshape((-1,))[accepts] = \
                    qslogl.reshape((-1,))[accepts]

                accepts = accepts.reshape((self.ntemps, self.nwalkers//2))

                self.nprop[:, jupdate::2] += 1.0
                self.nprop_accepted[:, jupdate::2] += accepts

            p, ratios = self._temperature_swaps(self._betas, p, logpost, logl)

            # TODO Should the notion of a "complete" iteration really include the temperature
            # adjustment?
            if adapt and self.ntemps > 1:
                dbetas = self._get_ladder_adjustment(self._time, self._betas, ratios)
                self._betas += dbetas
                logpost += self._tempered_likelihood(logl, betas=dbetas)

            if (self._time + 1) % thin == 0:
                if storechain:
                    self._chain[:, :, isave, :] = p
                    self._logposterior[:, :, isave] = logpost
                    self._loglikelihood[:, :, isave] = logl
                    self._beta_history[:, isave] = self._betas
                    isave += 1

            self._time += 1
            if swap_ratios:
                yield p, logpost, logl, ratios
            else:
                yield p, logpost, logl

    def _evaluate(self, ps):
        mapf = map if self.pool is None else self.pool.map
        results = list(mapf(self._likeprior, ps.reshape((-1, self.dim))))

        logl = np.fromiter((r[0] for r in results), np.float,
                           count=len(results)).reshape((self.ntemps, -1))
        logp = np.fromiter((r[1] for r in results), np.float,
                           count=len(results)).reshape((self.ntemps, -1))

        return logl, logp

    def _tempered_likelihood(self, logl, betas=None):
        """
        Compute tempered log likelihood.  This is usually a mundane multiplication, except for the
        special case where beta == 0 *and* we're outside the likelihood support.

        Here, we find a singularity that demands more careful attention; we allow the likelihood to
        dominate the temperature, since wandering outside the likelihood support causes a discontinuity.

        """

        if betas is None:
            betas = self._betas
        betas = betas.reshape((-1, 1))

        with np.errstate(invalid='ignore'):
            loglT = logl * betas
        loglT[np.isnan(loglT)] = -np.inf

        return loglT

    def _temperature_swaps(self, betas, p, logpost, logl):
        """
        Perform parallel-tempering temperature swaps on the state
        in ``p`` with associated ``logpost`` and ``logl``.

        """
        ntemps = len(betas)
        ratios = np.zeros(ntemps - 1)
        for i in range(ntemps - 1, 0, -1):
            bi = betas[i]
            bi1 = betas[i - 1]

            dbeta = bi1 - bi

            iperm = self._random.permutation(self.nwalkers)
            i1perm = self._random.permutation(self.nwalkers)

            raccept = np.log(self._random.uniform(size=self.nwalkers))
            paccept = dbeta * (logl[i, iperm] - logl[i - 1, i1perm])

            self.nswap[i] += self.nwalkers
            self.nswap[i - 1] += self.nwalkers

            asel = (paccept > raccept)
            nacc = np.sum(asel)

            self.nswap_accepted[i] += nacc
            self.nswap_accepted[i - 1] += nacc

            ratios[i - 1] = nacc / self.nwalkers

            ptemp = np.copy(p[i, iperm[asel], :])
            logltemp = np.copy(logl[i, iperm[asel]])
            logprtemp = np.copy(logpost[i, iperm[asel]])

            p[i, iperm[asel], :] = p[i - 1, i1perm[asel], :]
            logl[i, iperm[asel]] = logl[i - 1, i1perm[asel]]
            logpost[i, iperm[asel]] = logpost[i - 1, i1perm[asel]] \
                - dbeta * logl[i - 1, i1perm[asel]]

            p[i - 1, i1perm[asel], :] = ptemp
            logl[i - 1, i1perm[asel]] = logltemp
            logpost[i - 1, i1perm[asel]] = logprtemp + dbeta * logltemp

        return p, ratios

    def _get_ladder_adjustment(self, time, betas0, ratios):
        """
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.

        """

        betas = betas0.copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.adaptation_lag / (time + self.adaptation_lag)
        kappa = decay / self.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])

        # Don't mutate the ladder here; let the client code do that.
        return betas - betas0

    def _expand_chain(self, nsave):
        """
        Expand ``self._chain``, ``self._logposterior``,
        ``self._loglikelihood``, and ``self._beta_history``
        ahead of run to make room for new samples.

        :param nsave:
            The number of additional iterations for which to make room.

        :return ``isave``:
            Returns the index at which to begin inserting new entries.

        """

        if self._chain is None:
            isave = 0
            self._chain = np.zeros((self.ntemps, self.nwalkers, nsave,
                                    self.dim))
            self._logposterior = np.zeros((self.ntemps, self.nwalkers, nsave))
            self._loglikelihood = np.zeros((self.ntemps, self.nwalkers,
                                            nsave))
            self._beta_history = np.zeros((self.ntemps, nsave))
        else:
            isave = self._chain.shape[2]
            self._chain = np.concatenate((self._chain,
                                          np.zeros((self.ntemps,
                                                    self.nwalkers,
                                                    nsave, self.dim))),
                                         axis=2)
            self._logposterior = np.concatenate((self._logposterior,
                                                 np.zeros((self.ntemps,
                                                           self.nwalkers,
                                                           nsave))),
                                                axis=2)
            self._loglikelihood = np.concatenate((self._loglikelihood,
                                                  np.zeros((self.ntemps,
                                                           self.nwalkers,
                                                           nsave))),
                                                 axis=2)
            self._beta_history = np.concatenate((self._beta_history,
                                                 np.zeros((self.ntemps, nsave))),
                                                axis=1)

        return isave

    def log_evidence_estimate(self, logls=None, fburnin=0.1):
        """
        Thermodynamic integration estimate of the evidence for the sampler.

        :param logls: (optional) The log-likelihoods to use for
            computing the thermodynamic evidence.  If ``None`` (the
            default), use the stored log-likelihoods in the sampler.
            Should be of shape ``(Ntemps, Nwalkers, Nsamples)``.

        :param fburnin: (optional)
            The fraction of the chain to discard as burnin samples; only the
            final ``1-fburnin`` fraction of the samples will be used to
            compute the evidence; the default is ``fburnin = 0.1``.

        :return ``(logZ, dlogZ)``: Returns an estimate of the
            log-evidence and the error associated with the finite
            number of temperatures at which the posterior has been
            sampled.

        For details, see ``thermodynamic_integration_log_evidence``.
        """

        if logls is None:
            if self.loglikelihood is not None:
                logls = self.loglikelihood
            else:
                raise ValueError('No log likelihood values available.')

        istart = int(logls.shape[2] * fburnin + 0.5)
        mean_logls = np.mean(np.mean(logls, axis=1)[:, istart:], axis=1)

        return util.thermodynamic_integration_log_evidence(self._betas, mean_logls)

    @property
    def random(self):
        """
        Returns the random number generator for the sampler.

        """

        return self._random

    @property
    def betas(self):
        """
        Returns the current inverse temperature ladder of the sampler.

        """
        return self._betas

    @property
    def time(self):
        """
        Returns the current time, in iterations, of the sampler.

        """
        return self._time

    @property
    def chain(self):
        """
        Returns the stored chain of samples; shape ``(Ntemps,
        Nwalkers, Nsteps, Ndim)``.

        """
        return self._chain

    @property
    def flatchain(self):
        """Returns the stored chain, but flattened along the walker axis, so
        of shape ``(Ntemps, Nwalkers*Nsteps, Ndim)``.

        """

        s = self.chain.shape

        return self._chain.reshape((s[0], -1, s[3]))

    @property
    def logprobability(self):
        """
        Matrix of logprobability values; shape ``(Ntemps, Nwalkers, Nsteps)``.

        """
        return self._logposterior

    @property
    def loglikelihood(self):
        """
        Matrix of log-likelihood values; shape ``(Ntemps, Nwalkers, Nsteps)``.

        """
        return self._loglikelihood

    @property
    def beta_history(self):
        """
        Matrix of inverse temperatures; shape ``(Ntemps, Nsteps)``.

        """
        return self._beta_history

    @property
    def tswap_acceptance_fraction(self):
        """
        Returns an array of accepted temperature swap fractions for
        each temperature; shape ``(ntemps, )``.

        """
        return self.nswap_accepted / self.nswap

    @property
    def ntemps(self):
        """
        The number of temperature chains.

        """
        return len(self._betas)

    @property
    def acceptance_fraction(self):
        """
        Matrix of shape ``(Ntemps, Nwalkers)`` detailing the
        acceptance fraction for each walker.

        """
        return self.nprop_accepted / self.nprop

    @property
    def acor(self):
        """
        Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``.

        """
        return self.get_autocorr_time()

    def get_autocorr_time(self, window=50):
        """
        Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``.

        :param window: (optional)
            The size of the windowing function. This is equivalent to the
            maximum number of lags to use. (default: 50)

        """
        acors = np.zeros((self.ntemps, self.dim))

        for i in range(self.ntemps):
            x = np.mean(self._chain[i, :, :, :], axis=0)
            acors[i, :] = util.autocorr_integrated_time(x, window=window)
        return acors
