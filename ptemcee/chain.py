# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ['Chain']

import attr
import numpy as np

from . import util
from . import ensemble


@attr.s(slots=True)
class Chain(object):
    ensemble = attr.ib(type=ensemble.Ensemble)
    thin_by = attr.ib(type=int, default=None)
    _x = attr.ib(type=np.ndarray, init=False, default=None)
    _logP = attr.ib(type=np.ndarray, init=False, default=None)
    _logl = attr.ib(type=np.ndarray, init=False, default=None)
    _betas = attr.ib(type=np.ndarray, init=False, default=None)

    _swaps_proposed = attr.ib(type=np.ndarray, init=False)
    _swaps_accepted = attr.ib(type=np.ndarray, init=False)
    _jumps_proposed = attr.ib(type=np.ndarray, init=False)
    _jumps_accepted = attr.ib(type=np.ndarray, init=False)

    def __attrs_post_init__(self):
        if self.thin_by is None:
            self.thin_by = 1
        self._x = np.empty((0, self.ntemps, self.nwalkers, self.ndim), float)
        self._logP = np.empty((0, self.ntemps, self.nwalkers), float)
        self._logl = np.empty((0, self.ntemps, self.nwalkers), float)
        self._betas = np.empty((0, self.ntemps), float)

        self._jumps_proposed = np.zeros((self.ntemps, self.nwalkers))
        self._jumps_accepted = np.zeros((self.ntemps, self.nwalkers))
        self._swaps_proposed = np.zeros(self.ntemps - 1)
        self._swaps_accepted = np.zeros(self.ntemps - 1)

    @property
    def length(self):
        return self._x.shape[0]

    @property
    def time(self):
        return self.ensemble.time

    @property
    def ntemps(self):
        return self.ensemble.ntemps

    @property
    def nwalkers(self):
        return self.ensemble.nwalkers

    @property
    def ndim(self):
        return self.ensemble.ndim

    @property
    def jump_acceptance_ratio(self):
        return self._jumps_accepted / self._jumps_proposed

    @property
    def swap_acceptance_ratio(self):
        return self._swaps_accepted / self._swaps_proposed

    @staticmethod
    def _resize(array, count):
        shape = (count,) + array.shape[1:]
        return np.concatenate((array, np.empty(shape)), axis=0)

    def run(self, count):
        for _ in self.iterate(count):
            pass

    def iterate(self, count):
        self._x = self._resize(self._x, count)
        self._logP = self._resize(self._logP, count)
        self._logl = self._resize(self._logl, count)
        self._betas = self._resize(self._betas, count)
        for i in range(count):
            # TODO: off-by-one at start?
            for _ in range(self.thin_by):
                self.ensemble.step()

            self._x[i] = self.ensemble.x
            self._logP[i] = self.ensemble.logP
            self._logl[i] = self.ensemble.logl
            self._betas[i] = self.ensemble.betas
            self._swaps_proposed += self.ensemble.swaps_proposed
            self._swaps_accepted += self.ensemble.swaps_accepted
            self._jumps_proposed += self.ensemble.jumps_proposed
            self._jumps_accepted += self.ensemble.jumps_accepted
            yield self.ensemble

    def get_acts(self, window=50):
        acts = np.zeros((self.ntemps, self.ndim))

        for i in range(self.ntemps):
            x = np.mean(self._x[:, i, :, :], axis=1)
            acts[i, :] = util.get_integrated_act(x, window=window)
        return acts

    def log_evidence_estimate(self, fburnin=0.1):
        """
        Thermodynamic integration estimate of the evidence for the sampler.

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

        istart = int(self._logl.shape[0] * fburnin + 0.5)
        mean_logls = np.mean(np.mean(self._logl, axis=2)[istart:, :], axis=0)
        return util.thermodynamic_integration_log_evidence(self._betas[-1], mean_logls)
