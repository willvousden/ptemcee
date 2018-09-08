#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals

__all__ = ['_ladder', 'get_acf', 'get_integrated_act', 'thermodynamic_integration_log_evidence']

import numpy as np


def _ladder(betas):
    """
    Convert an arbitrary iterable of floats into a sorted numpy array.

    """

    betas = np.array(betas)
    betas[::-1].sort()
    return betas


def get_acf(x, axis=0, fast=False):
    """
    Estimate the autocorrelation function of a time series using the FFT.

    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.

    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.

    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    """
    x = np.atleast_1d(x)
    m = [slice(None), ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2 ** np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x - np.mean(x, axis=axis), n=2 * n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[tuple(m)].real
    m[axis] = 0
    return acf / acf[tuple(m)]


def get_integrated_act(x, axis=0, window=50, fast=False):
    """
    Estimate the integrated autocorrelation time of a time series.

    See `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ on
    MCMC and sample estimators for autocorrelation times.

    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.

    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.

    :param window: (optional)
        The size of the window to use. (default: 50)

    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    """
    # Compute the autocorrelation function.
    f = get_acf(x, axis=axis, fast=fast)

    # Special case 1D for simplicity.
    if len(f.shape) == 1:
        return 1 + 2 * np.sum(f[1:window])

    # N-dimensional case.
    m = [slice(None), ] * len(f.shape)
    m[axis] = slice(1, window)
    tau = 1 + 2 * np.sum(f[tuple(m)], axis=axis)

    return tau


def thermodynamic_integration_log_evidence(betas, logls):
    """
    Thermodynamic integration estimate of the evidence.

    :param betas: The inverse temperatures to use for the quadrature.

    :param logls:  The mean log-likelihoods corresponding to ``betas`` to use for
        computing the thermodynamic evidence.

    :return ``(logZ, dlogZ)``: Returns an estimate of the
        log-evidence and the error associated with the finite
        number of temperatures at which the posterior has been
        sampled.

    The evidence is the integral of the un-normalized posterior
    over all of parameter space:

    .. math::

        Z \\equiv \\int d\\theta \\, l(\\theta) p(\\theta)

    Thermodymanic integration is a technique for estimating the
    evidence integral using information from the chains at various
    temperatures.  Let

    .. math::

        Z(\\beta) = \\int d\\theta \\, l^\\beta(\\theta) p(\\theta)

    Then

    .. math::

        \\frac{d \\log Z}{d \\beta}
        = \\frac{1}{Z(\\beta)} \\int d\\theta l^\\beta p \\log l
        = \\left \\langle \\log l \\right \\rangle_\\beta

    so

    .. math::

        \\log Z(1) - \\log Z(0)
        = \\int_0^1 d\\beta \\left \\langle \\log l \\right\\rangle_\\beta

    By computing the average of the log-likelihood at the
    difference temperatures, the sampler can approximate the above
    integral.
    """
    if len(betas) != len(logls):
        raise ValueError('Need the same number of log(L) values as temperatures.')

    order = np.argsort(betas)[::-1]
    betas = betas[order]
    logls = logls[order]

    betas0 = np.copy(betas)
    if betas[-1] != 0:
        betas = np.concatenate((betas0, [0]))
        betas2 = np.concatenate((betas0[::2], [0]))

        # Duplicate mean log-likelihood of hottest chain as a best guess for beta = 0.
        logls2 = np.concatenate((logls[::2], [logls[-1]]))
        logls = np.concatenate((logls, [logls[-1]]))
    else:
        betas2 = np.concatenate((betas0[:-1:2], [0]))
        logls2 = np.concatenate((logls[:-1:2], [logls[-1]]))

    logZ = -np.trapz(logls, betas)
    logZ2 = -np.trapz(logls2, betas2)
    return logZ, np.abs(logZ - logZ2)
