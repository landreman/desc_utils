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
    filenames = [
        ".//tests//inputs//HELIOTRON_MJL.h5",
        ".//tests//inputs//HELIOTRON_MJL_current.h5",
    ]

    def test(eq, grid_type, kwargs, L, M, N, helicity_N):
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

    for filename in filenames:
        print("********* Processing file", filename, "*********")
        eq = desc.io.load(filename)
        for helicity_N in [0, eq.NFP, -eq.NFP]:
            results = []
            for grid_type, kwargs in zip(grid_types, kwargss):
                for L, M, N in zip(Ls, Ms, Ns):
                    results.append(test(eq, grid_type, kwargs, L, M, N, helicity_N))

            results = np.array(results)
            np.testing.assert_allclose(results, np.mean(results), rtol=1e-3)


def test_QA_QH():
    """
    The QA residual for QA should be small, and the QH residual for QH should be small
    """
    filenames = [
        ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5",
        ".//tests//inputs//LandremanPaul2022_QH_reactorScale_lowRes.h5",
    ]

    def test(eq, helicity_n):
        grid = QuadratureGrid(
            L=8,
            M=8,
            N=8,
            NFP=eq.NFP,
        )
        obj = ObjectiveFunction(
            QuasisymmetryTwoTermNormalized(
                grid=grid,
                helicity=(1, helicity_n * eq.NFP),
            ),
            eq,
        )
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(
            f"obj: {scalar_objective:11.9g}  file: {filename}  helicity_n: {helicity_n}"
        )
        return scalar_objective

    results = []
    for filename in filenames:
        eq = desc.io.load(filename)
        for helicity_n in [-1, 0, 1]:
            results.append(test(eq, helicity_n))

    results = np.array(results)
    print(results)

    # For the QA config, helicity = 0 should give the lowest residual:
    assert results[1] < results[0] * 1e-6
    assert results[1] < results[2] * 1e-6

    # For the QH config, helicity = -1 should give the lowest residual:
    assert results[3] < results[4] * 2e-5
    assert results[3] < results[5] * 2e-5

    # For the QA objective, the QA config should have lower residual than the QH config:
    assert results[1] < results[4] * 1e-7

    # For the QH objective, the QH config should have lower residual than the QA config:
    assert results[3] < results[0] * 2e-4
