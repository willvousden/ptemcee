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
    x = attr.ib(type=np.ndarray, init=False, default=None)
    logP = attr.ib(type=np.ndarray, init=False, default=None)
    logl = attr.ib(type=np.ndarray, init=False, default=None)
    betas = attr.ib(type=np.ndarray, init=False, default=None)

    swaps_proposed = attr.ib(type=np.ndarray, init=False)
    swaps_accepted = attr.ib(type=np.ndarray, init=False)
    jumps_proposed = attr.ib(type=np.ndarray, init=False)
    jumps_accepted = attr.ib(type=np.ndarray, init=False)

    def __attrs_post_init__(self):
        if self.thin_by is None:
            self.thin_by = 1
        self.x = np.empty((0, self.ntemps, self.nwalkers, self.ndim), float)
        self.logP = np.empty((0, self.ntemps, self.nwalkers), float)
        self.logl = np.empty((0, self.ntemps, self.nwalkers), float)
        self.betas = np.empty((0, self.ntemps), float)

        self.jumps_proposed = np.zeros((self.ntemps, self.nwalkers))
        self.jumps_accepted = np.zeros((self.ntemps, self.nwalkers))
        self.swaps_proposed = np.zeros(self.ntemps - 1)
        self.swaps_accepted = np.zeros(self.ntemps - 1)

    @property
    def length(self):
        return self.x.shape[0]

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
        return self.jumps_accepted / self.jumps_proposed

    @property
    def swap_acceptance_ratio(self):
        return self.swaps_accepted / self.swaps_proposed

    @staticmethod
    def _resize(array, count):
        shape = (count,) + array.shape[1:]
        return np.concatenate((array, np.empty(shape)), axis=0)

    def run(self, count):
        jp0 = self.jumps_proposed.copy()
        ja0 = self.jumps_accepted.copy()
        sp0 = self.swaps_proposed.copy()
        sa0 = self.swaps_accepted.copy()
        for _ in self.iterate(count):
            pass
        jp = self.jumps_proposed - jp0
        ja = self.jumps_accepted - ja0
        sp = self.swaps_proposed - sp0
        sa = self.swaps_accepted - sa0
        return ja / jp, sa / sp

    def iterate(self, count):
        start = self.length
        self.x = self._resize(self.x, count)
        self.logP = self._resize(self.logP, count)
        self.logl = self._resize(self.logl, count)
        self.betas = self._resize(self.betas, count)
        for i in range(start, start + count):
            for _ in range(self.thin_by):
                self.ensemble.step()
                self.swaps_proposed += self.ensemble.swaps_proposed
                self.swaps_accepted += self.ensemble.swaps_accepted
                self.jumps_proposed += self.ensemble.jumps_proposed
                self.jumps_accepted += self.ensemble.jumps_accepted

            self.x[i] = self.ensemble.x
            self.logP[i] = self.ensemble.logP
            self.logl[i] = self.ensemble.logl
            self.betas[i] = self.ensemble.betas
            yield self.ensemble

    def get_acts(self, window=50):
        acts = np.zeros((self.ntemps, self.ndim))

        for i in range(self.ntemps):
            x = np.mean(self.x[:, i, :, :], axis=1)
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

        istart = int(self.logl.shape[0] * fburnin + 0.5)
        mean_logls = np.mean(np.mean(self.logl, axis=2)[istart:, :], axis=0)
        return util.thermodynamic_integration_log_evidence(self.betas[-1], mean_logls)
