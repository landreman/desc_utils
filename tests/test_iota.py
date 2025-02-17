"""Tests for objectives related to iota."""

import numpy as np
import pytest

import desc.io
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives import *

from desc_utils import MeanIota, IotaAt


def test_mean_iota_resolution():
    """
    Confirm that the MeanIota objective function is
    approximately independent of grid resolution.
    """
    filenames = [
        ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5",
        ".//tests//inputs//HELIOTRON_MJL.h5",
    ]

    def test(eq, L, M, N):
        grid = QuadratureGrid(
            L=L,
            M=M,
            N=N,
            NFP=eq.NFP,
        )
        obj = ObjectiveFunction(
            MeanIota(
                grid=grid,
                target=0.6,
                eq=eq,
            ),
        )
        obj.build()
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(f"obj: {scalar_objective:11.9g}  L: {L}  M: {M}  N: {N}")
        assert obj.objectives[0]._coordinates == ""
        np.testing.assert_allclose(obj.objectives[0]._constants["quad_weights"], 1)
        return scalar_objective

    # Loop over grid resolutions:
    Ls = [32, 64, 32, 64, 32]
    Ms = [16, 16, 32, 32, 16]
    Ns = [16, 16, 16, 16, 32]

    for filename in filenames:
        print("********* Processing file", filename, "*********")
        eq = desc.io.load(filename)
        results = []
        for L, M, N in zip(Ls, Ms, Ns):
            results.append(test(eq, L, M, N))

        results = np.array(results)
        np.testing.assert_allclose(results, np.mean(results), rtol=1e-2)


def test_mean_iota_value():
    """ """
    filename = ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5"
    print(filename)
    eq = desc.io.load(filename)

    grid = QuadratureGrid(
        L=32,
        M=16,
        N=16,
        NFP=eq.NFP,
    )

    def test(target):
        obj = ObjectiveFunction(
            MeanIota(
                grid=grid,
                target=target,
                eq=eq,
            ),
        )
        obj.build()
        scalar_objective = obj.compute_scalar(obj.x(eq))

        expected = 0.5 * (0.42 - target) ** 2
        rel_diff = abs(
            (scalar_objective - expected) / (0.5 * (scalar_objective + expected))
        )
        print(
            f"target: {target}  obj: {scalar_objective:11.9g}  "
            f"expected: {expected}  rel diff: {rel_diff}"
        )
        np.testing.assert_allclose(scalar_objective, expected, rtol=1e-2)
        assert obj.objectives[0]._coordinates == ""
        np.testing.assert_allclose(obj.objectives[0]._constants["quad_weights"], 1)

        return scalar_objective

    targets = [-0.6, 0, 0.7]
    for target in targets:
        test(target)


def test_iota_at_resolution():
    """
    Confirm that the IotaAt objective function is
    approximately independent of grid resolution.
    """
    filenames = [
        ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5",
        ".//tests//inputs//HELIOTRON_MJL.h5",
    ]

    def test(eq, M, N):
        grid = LinearGrid(
            rho=0.6,
            M=M,
            N=N,
            NFP=eq.NFP,
        )
        obj = ObjectiveFunction(
            IotaAt(
                grid=grid,
                target=0.6,
                eq=eq,
            ),
        )
        obj.build()
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(f"obj: {scalar_objective:11.9g}  M: {M}  N: {N}")
        assert obj.objectives[0]._coordinates == ""
        np.testing.assert_allclose(obj.objectives[0]._constants["quad_weights"], 1)
        return scalar_objective

    # Loop over grid resolutions:
    # Ls = [8, 16, 8, 16, 8]
    # Ms = [8, 8, 16, 16, 8]
    # Ns = [8, 8, 8, 8, 16]

    Ms = [16, 32, 16]
    Ns = [16, 16, 32]

    for filename in filenames:
        print("********* Processing file", filename, "*********")
        eq = desc.io.load(filename)
        results = []
        for M, N in zip(Ms, Ns):
            results.append(test(eq, M, N))

        results = np.array(results)
        np.testing.assert_allclose(results, np.mean(results), rtol=1e-2)


def test_iota_at_value():
    """ """
    filename = ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5"
    print(filename)
    eq = desc.io.load(filename)

    grid = LinearGrid(
        rho=0.5,
        M=16,
        N=16,
        NFP=eq.NFP,
    )

    def test(target):
        obj = ObjectiveFunction(
            IotaAt(
                grid=grid,
                target=target,
                eq=eq,
            ),
        )
        obj.build()
        scalar_objective = obj.compute_scalar(obj.x(eq))

        expected = 0.5 * (0.42 - target) ** 2
        rel_diff = abs(
            (scalar_objective - expected) / (0.5 * (scalar_objective + expected))
        )
        print(
            f"target: {target}  obj: {scalar_objective:11.9g}  "
            f"expected: {expected}  rel diff: {rel_diff}"
        )
        np.testing.assert_allclose(scalar_objective, expected, rtol=1e-2)
        assert obj.objectives[0]._coordinates == ""
        np.testing.assert_allclose(obj.objectives[0]._constants["quad_weights"], 1)

        return scalar_objective

    targets = [-0.6, 0, 0.7]
    for target in targets:
        test(target)
