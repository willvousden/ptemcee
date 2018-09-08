# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['Chain', 'ChainIterator']

import numpy as np
#  import multiprocessing
import attr

from attr import Factory
from attr.validators import instance_of, optional
from np.random.mtrand import RandomState

from . import util
from .sampler import Sampler


@attr.s(slots=True)
class ChainIterator(object):
    _config = attr.ib(validator=instance_of(Configuration))

    # Initial walker positions and probabilities.
    x = attr.ib(converter=np.array)
    logP = attr.ib(default=None)
    logl = attr.ib(default=None)

    thin_by = attr.ib(converter=int, default=1)
    @thin_by.validator
    def _is_positive(self, attribute, value):
        if value < 1:
            raise ValueError('{} must be positive.'.format(attribute.name))

    betas = attr.ib(converter=_ladder)
    @betas.validator
    def _is_consistent(self, attribute, value):
        if len(value) != len(self.x):
            raise ValueError('Number of temperatures not consistent with '
                             'starting positions.')

    _random = attr.ib(validator=instance_of(RandomState), default=Factory(RandomState))
    _map = attr.ib(validator=is_callable, default=map)

    time = attr.ib(init=False, default=0)
    nwalkers = attr.ib(init=False)
    ntemps = attr.ib(init=False)
    ndim = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.ntemps, self.nwalkers, self.ndim = self.x.shape

        # If we have no likelihood or prior values, compute them.
        if self.logP is None or self.logl is None:
            logl, logp = self._evaluate(self.x)
            self.logP = self._tempered_likelihood(logl) + logp
            self.logl = logl

        if (self.logP == -np.inf).any():
            raise ValueError('Attempting to start with samples outside '
                             'posterior support.')

    def __iter__(self):
        return self

    def __next__(self):
        for i in range(self.thin_by):
            self._stretch(self.x, self.logP, self.logl)
            self.x, ratios = self._temperature_swaps(self.betas,
                                                     self.x,
                                                     self.logP,
                                                     self.logl)

            # TODO Should the notion of a 'complete' iteration really include
            # the temperature adjustment?
            if adapt and self.ntemps > 1:
                dbetas = self._get_ladder_adjustment(self.time,
                                                     self.betas,
                                                     ratios)
                self.betas += dbetas
                self.logP += self._tempered_likelihood(self.logl, betas=dbetas)

            self.time += 1

        yield self.x, self.logP, self.logl

    def _stretch(self, x, logP, logl):
        '''
        Perform the stretch-move proposal on each ensemble.

        '''

        w = self.nwalkers // 2
        d = self.ndim
        t = self.ntemps
        a = self._config.scale_factor

        for j in [0, 1]:
            # Get positions of walkers to be updated and walker to be sampled.
            j_update = j
            j_sample = (j + 1) % 2
            x_update = x[:, j_update::2, :]
            x_sample = x[:, j_sample::2, :]

            z = np.exp(self._random.uniform(low=-np.log(a),
                                            high=np.log(a),
                                            size=(t, w)))

            y = np.empty((t, w, d))
            for k in range(t):
                js = self._random.randint(0, high=w, size=w)
                y[k, :, :] = x_sample[k, js, :] + \
                             z[k, :].reshape((w, 1)) * \
                             (x_update[k, :, :] - x_sample[k, js, :])

            y_logl, y_logp = self._evaluate(y)
            y_logP = self._tempered_likelihood(y_logl) + y_logp

            logp_accept = d * np.log(z) + y_logP - logP[:, j_update::2]
            logr = np.log(self._random.uniform(low=0, high=1,
                                               size=(t, w)))

            accepts = logr < logp_accept
            accepts = accepts.flatten()

            x_update.reshape((-1, d))[accepts, :] = \
                y.reshape((-1, d))[accepts, :]
            logP[:, j_update::2].reshape((-1,))[accepts] = \
                y_logP.reshape((-1,))[accepts]
            logl[:, j_update::2].reshape((-1,))[accepts] = \
                y_logl.reshape((-1,))[accepts]

            accepts = accepts.reshape((t, w))

            # TODO
            # self.nprop[:, j_update::2] += 1.0
            # self.nprop_accepted[:, j_update::2] += accepts

    def _evaluate(self, x):
        '''
        Evaluate the log likelihood and log prior functions at the specified walker positions.

        '''

        results = list(self._map(self._config.evaluator,
                                    x.reshape((-1, self.ndim))))
        return tuple(np.array(results).T)

    def _tempered_likelihood(self, logl, betas=None):
        '''
        Compute tempered log likelihood.  This is usually a mundane
        multiplication, except for the special case where beta == 0 *and* we're
        outside the likelihood support.

        Here, we find a singularity that demands more careful attention; we
        allow the likelihood to dominate the temperature, since wandering
        outside the likelihood support causes a discontinuity.

        '''

        if betas is None:
            betas = self.betas

        with np.errstate(invalid='ignore'):
            loglT = logl * betas[:, None]
        loglT[np.isnan(loglT)] = -np.inf

        return loglT

    def _get_ladder_adjustment(self, time, betas0, ratios):
        '''
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.

        '''

        betas = betas0.copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self._config.adaptation_lag / (time + self._config.adaptation_lag)
        kappa = decay / self._config.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])

        # Don't mutate the ladder here; let the client code do that.
        return betas - betas0

    def _temperature_swaps(self, x, logP, logl):
        '''
        Perform parallel-tempering temperature swaps on the state
        in ``x`` with associated ``logP`` and ``logl``.

        '''

        ntemps = len(self.betas)
        ratios = np.zeros(ntemps - 1)
        for i in range(ntemps - 1, 0, -1):
            bi = self.betas[i]
            bi1 = self.betas[i - 1]

            dbeta = bi1 - bi

            iperm = self._random.permutation(self.nwalkers)
            i1perm = self._random.permutation(self.nwalkers)

            raccept = np.log(self._random.uniform(size=self.nwalkers))
            paccept = dbeta * (logl[i, iperm] - logl[i - 1, i1perm])

            # TODO
            # self.nswap[i] += self.nwalkers
            # self.nswap[i - 1] += self.nwalkers

            # How many swaps were accepted?
            sel = (paccept > raccept)
            n = np.sum(sel)

            # TODO
            # self.nswap_accepted[i] += n
            # self.nswap_accepted[i - 1] += n

            ratios[i - 1] = n / self.nwalkers

            x_temp = np.copy(x[i, iperm[sel], :])
            logl_temp = np.copy(logl[i, iperm[sel]])
            logP_temp = np.copy(logP[i, iperm[sel]])

            x[i, iperm[sel], :] = x[i - 1, i1perm[sel], :]
            logl[i, iperm[sel]] = logl[i - 1, i1perm[sel]]
            logP[i, iperm[sel]] = logP[i - 1, i1perm[sel]] \
                - dbeta * logl[i - 1, i1perm[sel]]

            x[i - 1, i1perm[sel], :] = x_temp
            logl[i - 1, i1perm[sel]] = logl_temp
            logP[i - 1, i1perm[sel]] = logP_temp + dbeta * logl_temp

        return x, ratios
