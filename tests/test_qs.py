"""Tests for quasisymmetry objective."""

import numpy as np
import pytest

import desc.io
from desc.grid import LinearGrid, QuadratureGrid
from desc.optimize import Optimizer
from desc.objectives import *

from desc_utils import QuasisymmetryTwoTermNormalized


def test_quasisymmetry_resolution():
    """
    Confirm that the quasisymmetry objective function is
    approximately independent of grid resolution.
    """
    filename = ".//tests//inputs//HELIOTRON_MJL.h5"
    eq = desc.io.load(filename)

    def test(grid_type, kwargs, L, M, N, helicity_N):
        grid = grid_type(
            L=L,
            M=M,
            N=N,
            NFP=eq.NFP,
            **kwargs,
        )
        obj = ObjectiveFunction(
            QuasisymmetryTwoTermNormalized(
                grid=grid,
                helicity=(1, helicity_N),
            ),
            eq,
        )
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(
            f"obj: {scalar_objective:11.9g}  helicity: {helicity_N}  grid: {grid_type}  L: {L}  M: {M}  N: {N}"
        )
        return scalar_objective

    # Loop over grid trypes. For LinearGrid we need to drop the point at rho=0 to avoid a divide-by-0
    # grid_types = [LinearGrid, QuadratureGrid]
    # kwargss = [{"axis": False}, {}]
    # kwargss = [{}, {}]

    grid_types = [QuadratureGrid]
    kwargss = [{}]

    # Loop over grid resolutions:
    Ls = [8, 16, 8, 16, 8]
    Ms = [8, 8, 16, 16, 8]
    Ns = [8, 8, 8, 8, 16]

    # Ls = [16, 32, 16, 16]
    # Ms = [16, 16, 32, 16]
    # Ns = [16, 16, 16, 32]

    for helicity_N in [0, eq.NFP, -eq.NFP]:
        results = []
        for grid_type, kwargs in zip(grid_types, kwargss):
            for L, M, N in zip(Ls, Ms, Ns):
                results.append(test(grid_type, kwargs, L, M, N, helicity_N))

        results = np.array(results)
        np.testing.assert_allclose(results, np.mean(results), rtol=1e-4)