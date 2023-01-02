"""Tests for magnetic well objective."""

import numpy as np
import pytest

import desc.io
from desc.grid import LinearGrid, QuadratureGrid
from desc.optimize import Optimizer
from desc.objectives import *

from desc_utils import MagneticWellThreshold


def test_magwell_resolution():
    """
    Confirm that the magnetic well objective function is
    approximately independent of grid resolution.
    """
    filenames = [
        ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5",
        ".//tests//inputs//HELIOTRON_MJL.h5",
    ]

    def test(eq, grid_type, kwargs, L, M, N):
        grid = grid_type(
            L=L,
            M=M,
            N=N,
            NFP=eq.NFP,
            **kwargs,
        )
        obj = ObjectiveFunction(
            MagneticWellThreshold(
                grid=grid,
                threshold=-0.1,
            ),
            eq,
        )
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(
            f"obj: {scalar_objective:11.9g}  grid: {grid_type}  L: {L}  M: {M}  N: {N}"
        )
        return scalar_objective

    # Loop over grid trypes. For LinearGrid we need to drop the point at rho=0 to avoid a divide-by-0
    # grid_types = [LinearGrid, QuadratureGrid]
    # kwargss = [{"axis": False}, {}]
    # kwargss = [{}, {}]

    grid_types = [QuadratureGrid]
    kwargss = [{}]

    # Loop over grid resolutions:
    # Ls = [8, 16, 8, 16, 8]
    # Ms = [8, 8, 16, 16, 8]
    # Ns = [8, 8, 8, 8, 16]

    # Ls = [16, 32, 16, 32, 16]
    # Ms = [8, 8, 16, 16, 8]
    # Ns = [8, 8, 8, 8, 16]

    Ls = [16, 32, 16, 32, 16]
    Ms = [16, 16, 32, 32, 16]
    Ns = [16, 16, 16, 16, 32]

    for filename in filenames:
        print("********* Processing file", filename, "*********")
        eq = desc.io.load(filename)
        results = []
        for grid_type, kwargs in zip(grid_types, kwargss):
            for L, M, N in zip(Ls, Ms, Ns):
                results.append(test(eq, grid_type, kwargs, L, M, N))

        results = np.array(results)
        np.testing.assert_allclose(results, np.mean(results), rtol=0.002)


def test_magwell_value():
    """
    Confirm the specific value of the magnetic well objective function.

    For the Landreman & Paul QA, (d^2 V / d s^2) / V is ~ 0.03 at all radii.
    """
    filename = ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5"

    eq = desc.io.load(filename)
    grid = QuadratureGrid(
        L=8,
        M=8,
        N=8,
        NFP=eq.NFP,
    )

    def test(threshold):
        obj = ObjectiveFunction(
            MagneticWellThreshold(
                grid=grid,
                threshold=threshold,
            ),
            eq,
        )
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(f"obj: {scalar_objective:11.9g}  threshold: {threshold}")

        if threshold > 0.03:
            assert scalar_objective < 1e-15
        else:
            expected = 0.5 * (0.03 - threshold) ** 2
            rel_diff = abs(
                (scalar_objective - expected) / (0.5 * (scalar_objective + threshold))
            )
            print(f"  expected: {expected}  rel diff: {rel_diff}")
            np.testing.assert_allclose(scalar_objective, expected, rtol=0.013)

        return scalar_objective

    thresholds = [0, 0.06, -0.06, 0.12, -0.12]
    for threshold in thresholds:
        test(threshold)


def test_independent_of_size_and_B():
    """
    The magnetic well objective should be unchanged under scaling the size or field strength of a configuration
    """
    filenames = [
        ".//tests//inputs//circular_model_tokamak_output.h5",
        ".//tests//inputs//circular_model_tokamak_2xB_output.h5",
        ".//tests//inputs//circular_model_tokamak_2xSize_output.h5",
    ]

    def test(eq):
        grid = QuadratureGrid(
            L=eq.L,
            M=eq.M,
            N=eq.M,  # Note M instead of N here
            NFP=eq.NFP,
        )
        obj = ObjectiveFunction(
            MagneticWellThreshold(
                grid=grid,
                threshold=-0.3,
            ),
            eq,
        )
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(f"obj: {scalar_objective:11.9g}  file: {filename}")
        assert np.abs(scalar_objective) > 0.01
        return scalar_objective

    results = []
    for filename in filenames:
        eq = desc.io.load(filename)[-1]
        results.append(test(eq))

    results = np.array(results)
    print(results)

    # Results should all be the same:
    np.testing.assert_allclose(results, np.mean(results), rtol=1e-4)
