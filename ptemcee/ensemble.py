import attr
import numpy as np
import itertools

from attr.validators import instance_of
from numpy.random.mtrand import RandomState

from . import util

__all__ = ['Ensemble', 'EnsembleConfiguration']


@attr.s(slots=True, frozen=True)
class EnsembleConfiguration(object):
    adaptation_lag = attr.ib()
    adaptation_time = attr.ib()
    scale_factor = attr.ib()
    evaluator = attr.ib()


@attr.s(slots=True)
class Ensemble(object):
    """
    This contains as little contextual information as it can.  It represents an ensemble.py that performs steps in the
    parameter space.

    """

    _config = attr.ib(type=EnsembleConfiguration, validator=instance_of(EnsembleConfiguration))

    betas = attr.ib(type=np.ndarray, converter=util._ladder)

    # Initial walker positions and probabilities.
    x = attr.ib(type=np.ndarray, converter=np.array)
    logP = attr.ib(type=np.ndarray, default=None)
    logl = attr.ib(type=np.ndarray, default=None)

    adaptive = attr.ib(type=bool, converter=bool, default=False)

    _random = attr.ib(type=RandomState, validator=instance_of(RandomState), factory=RandomState)
    _mapper = attr.ib(default=map)

    time = attr.ib(type=int, init=False, default=0)
    nwalkers = attr.ib(type=int, init=False)
    ntemps = attr.ib(type=int, init=False)
    ndim = attr.ib(type=int, init=False)

    jumps_proposed = attr.ib(type=np.ndarray, init=False, default=None)
    jumps_accepted = attr.ib(type=np.ndarray, init=False, default=None)
    swaps_proposed = attr.ib(type=np.ndarray, init=False, default=None)
    swaps_accepted = attr.ib(type=np.ndarray, init=False, default=None)

    @_mapper.validator
    def _is_callable(self, attribute, value):
        if not callable(value):
            raise ValueError('{} must be callable.'.format(attribute.name))

    @betas.validator
    def _is_consistent(self, attribute, value):
        if len(value) != len(self.x):
            raise ValueError('Number of temperatures not consistent with starting positions.')

    def __attrs_post_init__(self):
        self.ntemps, self.nwalkers, self.ndim = self.x.shape

        self.jumps_proposed = np.ones((self.ntemps, self.nwalkers))
        self.swaps_proposed = np.full(self.ntemps - 1, self.nwalkers)

        # If we have no likelihood or prior values, compute them.
        if self.logP is None or self.logl is None:
            logl, logp = self._evaluate(self.x)
            self.logP = self._tempered_likelihood(logl) + logp
            self.logl = logl

        if (self.logP == -np.inf).any():
            raise ValueError('Attempting to start with samples outside posterior support.')

    def step(self):
        self._stretch(self.x, self.logP, self.logl)
        self.x = self._temperature_swaps(self.x, self.logP, self.logl)
        ratios = self.swaps_accepted / self.swaps_proposed

        # TODO: Should the notion of a 'complete' iteration really include the temperature adjustment?
        if self.adaptive and self.ntemps > 1:
            dbetas = self._get_ladder_adjustment(self.time,
                                                 self.betas,
                                                 ratios)
            self.betas += dbetas
            self.logP += self._tempered_likelihood(self.logl, betas=dbetas)

        self.time += 1

    def _stretch(self, x, logP, logl):
        """
        Perform the stretch-move proposal on each ensemble.py.

        """

        self.jumps_accepted = np.zeros((self.ntemps, self.nwalkers))
        w = self.nwalkers // 2
        d = self.ndim
        t = self.ntemps
        loga = np.log(self._config.scale_factor)

        for j in [0, 1]:
            # Get positions of walkers to be updated and walker to be sampled.
            j_update = j
            j_sample = (j + 1) % 2
            x_update = x[:, j_update::2, :]
            x_sample = x[:, j_sample::2, :]

            z = np.exp(self._random.uniform(low=-loga, high=loga, size=(t, w)))
            y = np.empty((t, w, d))
            for k in range(t):
                js = self._random.randint(0, high=w, size=w)
                y[k, :, :] = (x_sample[k, js, :] +
                              z[k, :].reshape((w, 1)) *
                              (x_update[k, :, :] - x_sample[k, js, :]))

            y_logl, y_logp = self._evaluate(y)
            y_logP = self._tempered_likelihood(y_logl) + y_logp

            logp_accept = d * np.log(z) + y_logP - logP[:, j_update::2]
            logr = np.log(self._random.uniform(low=0, high=1, size=(t, w)))

            accepts = logr < logp_accept
            accepts = accepts.flatten()

            x_update.reshape((-1, d))[accepts, :] = y.reshape((-1, d))[accepts, :]
            logP[:, j_update::2].reshape((-1,))[accepts] = y_logP.reshape((-1,))[accepts]
            logl[:, j_update::2].reshape((-1,))[accepts] = y_logl.reshape((-1,))[accepts]

            self.jumps_accepted[:, j_update::2] = accepts.reshape((t, w))

    def _evaluate(self, x):
        """
        Evaluate the log likelihood and log prior functions at the specified walker positions.

        """

        # Make a flattened iterable of the results, of alternating logL and logp.
        shape = x.shape[:-1]
        values = x.reshape((-1, self.ndim))
        length = len(values)
        results = itertools.chain.from_iterable(self._mapper(self._config.evaluator, values))

        # Construct into a pre-allocated ndarray.
        array = np.fromiter(results, float, 2 * length).reshape(shape + (2,))
        return tuple(np.rollaxis(array, -1))

    def _tempered_likelihood(self, logl, betas=None):
        """
        Compute tempered log likelihood.  This is usually a mundane multiplication, except for the special case where
        beta == 0 *and* we're outside the likelihood support.

        Here, we find a singularity that demands more careful attention; we allow the likelihood to dominate the
        temperature, since wandering outside the likelihood support causes a discontinuity.

        """

        if betas is None:
            betas = self.betas

        with np.errstate(invalid='ignore'):
            loglT = logl * betas[:, None]
        loglT[np.isnan(loglT)] = -np.inf

        return loglT

    def _get_ladder_adjustment(self, time, betas0, ratios):
        """
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.

        """

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
        """
        Perform parallel-tempering temperature swaps on the state in ``x`` with associated ``logP`` and ``logl``.

        """

        nwalkers = self.nwalkers
        ntemps = len(self.betas)
        self.swaps_accepted = np.empty(ntemps - 1)
        for i in range(ntemps - 1, 0, -1):
            bi = self.betas[i]
            bi1 = self.betas[i - 1]

            dbeta = bi1 - bi

            iperm = self._random.permutation(nwalkers)
            i1perm = self._random.permutation(nwalkers)

            raccept = np.log(self._random.uniform(size=nwalkers))
            paccept = dbeta * (logl[i, iperm] - logl[i - 1, i1perm])

            # How many swaps were accepted?
            sel = (paccept > raccept)
            self.swaps_accepted[i - 1] = np.sum(sel)

            x_temp = np.copy(x[i, iperm[sel], :])
            logl_temp = np.copy(logl[i, iperm[sel]])
            logP_temp = np.copy(logP[i, iperm[sel]])

            x[i, iperm[sel], :] = x[i - 1, i1perm[sel], :]
            logl[i, iperm[sel]] = logl[i - 1, i1perm[sel]]
            logP[i, iperm[sel]] = logP[i - 1, i1perm[sel]] - dbeta * logl[i - 1, i1perm[sel]]

            x[i - 1, i1perm[sel], :] = x_temp
            logl[i - 1, i1perm[sel]] = logl_temp
            logP[i - 1, i1perm[sel]] = logP_temp + dbeta * logl_temp

        return x
