# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ['make_ladder', 'Sampler']

import operator
import attr
import itertools
import numpy as np

from numpy.random.mtrand import RandomState

from . import util, chain, ensemble


def make_ladder(ndim, ntemps=None, Tmax=None):
    """
    Returns a ladder of :math:`\\beta \\equiv 1/T` under a geometric spacing that is determined by the
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
            raise ValueError('Must specify at least one of ``ntemps'' and '
                             'finite ``Tmax``.')

        # Determine ntemps from Tmax.
        ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

    betas = np.logspace(0, -np.log10(Tmax), ntemps)
    if appendInf:
        # Use a geometric spacing, but replace the top-most temperature with
        # infinity.
        betas = np.concatenate((betas, [0]))

    return betas


@attr.s(slots=True, frozen=True)
class LikePriorEvaluator(object):
    """
    Wrapper class for logl and logp.

    """

    logl = attr.ib()
    logp = attr.ib()
    logl_args = attr.ib(factory=list)
    logp_args = attr.ib(factory=list)
    logl_kwargs = attr.ib(factory=dict)
    logp_kwargs = attr.ib(factory=dict)

    def __call__(self, x):
        lp = self.logp(x, *self.logp_args, **self.logp_kwargs)
        if np.isnan(lp):
            raise ValueError('Prior function returned NaN.')

        if lp == float('-inf'):
            # Can't return -inf, since this messes with beta=0 behaviour.
            ll = 0
        else:
            ll = self.logl(x, *self.logl_args, **self.logl_kwargs)
            if np.isnan(ll).any():
                raise ValueError('Log likelihood function returned NaN.')

        return ll, lp


@attr.s(slots=True, frozen=True)
class Sampler(object):
    # Mandatory parameters.
    nwalkers = attr.ib(converter=int)
    ndim = attr.ib(converter=int)

    logl = attr.ib()
    logp = attr.ib()
    logl_args = attr.ib(converter=list, factory=list)
    logp_args = attr.ib(converter=list, factory=list)
    logl_kwargs = attr.ib(converter=dict, factory=dict)
    logp_kwargs = attr.ib(converter=dict, factory=dict)

    betas = attr.ib(default=None)

    # Tuning parameters.
    adaptive = attr.ib(converter=bool, default=False)
    adaptation_lag = attr.ib(converter=int, default=10000)
    adaptation_time = attr.ib(converter=int, default=100)
    scale_factor = attr.ib(converter=float, default=2)

    _mapper = attr.ib(default=map)
    _evaluator = attr.ib(type=LikePriorEvaluator, init=False, default=None)
    _data = attr.ib(type=np.ndarray, init=False, default=None)

    @nwalkers.validator
    def _validate_nwalkers(self, attribute, value):
        if value % 2 != 0:
            raise ValueError('The number of walkers must be even.')
        if value < 2 * self.ndim:
            raise ValueError('The number of walkers must be greater than 2 * dimension.')

    @ndim.validator
    def _validate_ndim(self, attribute, value):
        if value < 1:
            raise ValueError('Number of dimensions must be positive.')

    @betas.validator
    def _validate_betas(self, attribute, value):
        try:
            # see if betas is an integer.
            v = operator.index(value)
            if v < 1:
                raise ValueError("Need at least one temperature")
            return
        except TypeError:
            pass

        if len(value) < 1:
            raise ValueError('Need at least one temperature!')
        if (value < 0).any():
            raise ValueError('Temperatures must be non-negative.')

    @logl.validator
    @logp.validator
    def _is_callable(self, attribute, value):
        if not callable(value):
            raise TypeError('{} must be callable'.format(attribute.name))

    def __attrs_post_init__(self):
        if self.betas is None:
            betas = make_ladder(self.ndim)
        elif isinstance(self.betas, int):
            # Treat this as the number of temperatures to use.
            betas = make_ladder(self.ndim, self.betas)
        else:
            betas = util._ladder(self.betas)

        object.__setattr__(self, 'betas', betas)
        object.__setattr__(self, '_evaluator',
                           LikePriorEvaluator(logl=self.logl,
                                              logp=self.logp,
                                              logl_args=self.logl_args,
                                              logp_args=self.logp_args,
                                              logl_kwargs=self.logl_kwargs,
                                              logp_kwargs=self.logp_kwargs))

    def ensemble(self, x, random=None):
        if random is None:
            random = RandomState()
        elif not isinstance(random, RandomState):
            raise TypeError('Invalid random state.')

        config = ensemble.EnsembleConfiguration(adaptation_lag=self.adaptation_lag,
                                                adaptation_time=self.adaptation_time,
                                                scale_factor=self.scale_factor,
                                                evaluator=self._evaluator)
        return ensemble.Ensemble(x=x,
                                 betas=self.betas.copy(),
                                 config=config,
                                 adaptive=self.adaptive,
                                 random=random,
                                 mapper=self._mapper)

    def sample(self, x, random=None, thin_by=None):
        """
        Return a stateless iterator.

        """

        if thin_by is None:
            thin_by = 1

        # Don't yield the starting state.
        ensemble = self.ensemble(x, random)
        while True:
            for _ in range(thin_by):
                ensemble.step()
            yield ensemble

    def chain(self, x, random=None, thin_by=None):
        """
        Create a stateful chain that stores its history.

        """
        return chain.Chain(self.ensemble(x, random), thin_by)
