"""Tests for shape objectives."""

import numpy as np
import pytest

import desc.io
from desc.grid import LinearGrid, QuadratureGrid
from desc.optimize import Optimizer
from desc.objectives import *

from desc_utils import AxisymmetryBarrier


def test_axisymmetry_barrier_value():
    """
    Confirm the specific value of the axisymmetry barrier objective function.
    """
    filename = ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5"
    R001 = 1.7455474614112452
    Z001 = 1.450812602359646

    R_threshold = 2.2
    Z_threshold = 1.9

    eq = desc.io.load(filename)

    def test(R_threshold, Z_threshold):
        obj = ObjectiveFunction(
            AxisymmetryBarrier(
                R_threshold=R_threshold,
                Z_threshold=Z_threshold,
            ),
            eq,
        )
        scalar_objective = obj.compute_scalar(obj.x(eq))
        residual1 = 0
        residual2 = 0
        if R001 < R_threshold:
            residual1 = R_threshold - R001
        if Z001 < Z_threshold:
            residual2 = Z_threshold - Z001
        should_be = 0.5 * (residual1**2 + residual2**2)
        print(
            f"R_threshold: {R_threshold}  Z_threshold: {Z_threshold}"
            f"  obj: {scalar_objective:11.9g}  should be: {should_be}"
        )

        np.testing.assert_allclose(scalar_objective, should_be)

    for R_threshold in [0, 0.5, 2.2]:
        for Z_threshold in [0, 0.6, 1.9]:
            test(R_threshold, Z_threshold)
