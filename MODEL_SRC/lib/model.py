#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of low dimensional system with known governing equations."""

import sys
import numpy as np

sys.dont_write_bytecode = True

def F_simple_2d_system(x):
    """Simple toy problem with known Koopman eigenfunctions in `Lusch paper`_.

    .. warning::

        Returned Jacobian is not used any more.

    .. _Lusch paper: https://www.nature.com/articles/s41467-018-07210-0/

    Note:

        In this example, :math:`\mu` = -0.05, :math:`\lambda` = -1.0.

    Args:

        x (:obj:`numpy.ndarray`): system state.

    Returns:

        :obj:`tuple` : A tuple of time derivative and Jacobian.

    """

    # We setup the model parameters then simply evaluate.

    mu = -0.05
    plambda = -1.0

    # Evaluate time derivative

    F = np.zeros(x.shape)
    F[0] = mu*x[0]
    F[1] = plambda*(x[1] - x[0]**2)

    # Evaluate Jacobian

    JF = np.zeros((2, 2))
    JF[0, 0] = mu
    JF[0, 1] = 0
    JF[1, 0] = -2.0*plambda*x[0]
    JF[1, 1] = plambda

    return F, JF


def F_duffing_2d_system(x):
    """Simple toy problem with multiple attractors in `Otto's paper`_.

    .. _Otto's paper: https://arxiv.org/abs/1712.01378

    .. warning::

        Returned Jacobian is not used any more.

    Note:

        In this example, :math:`delta=0.5`, :math:`beta=-1.0`, :math:`alpha=1.0`.

    Args:

        x (:obj:`numpy.ndarray`): system state.

    Returns:

        :obj:`tuple` : A tuple of time derivative and Jacobian.

    """

    # We setup the model parameters then simply evaluate.
    delta = 0.5
    beta  = -1.0
    alpha = 1.0

    # Evaluate on the time derivative

    F = np.zeros(x.shape)
    F[0] = x[1]
    F[1] = -delta * x[1] - x[0] * (beta + alpha * x[0]**2)

    # Evaluate on the Jacobian

    JF = np.zeros((2,2))
    JF[0, 0] = 0
    JF[0, 1] = 1
    JF[1, 0] = -beta - 3.0*alpha*x[0]**2
    JF[1, 1] = -delta

    return F, JF