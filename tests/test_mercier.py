"""Tests for Mercier stability objective."""

import numpy as np
import pytest

import desc.io
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives import *

from desc_utils import Mercier_normalization, MercierThreshold


def test_mercier_resolution():
    """
    Confirm that the Mercier objective function is
    approximately independent of grid resolution.
    """
    filenames = [
        ".//tests//inputs//circular_model_tokamak_finiteBeta_output.h5",
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
            MercierThreshold(
                grid=grid,
                threshold=0.6,
                eq=eq,
            ),
        )
        obj.build()
        obj.print_value(obj.x(eq))
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(
            f"obj: {scalar_objective:11.9g}  grid: {grid_type}  L: {L}  M: {M}  N: {N}"
        )
        return scalar_objective

    grid_types = [QuadratureGrid]
    kwargss = [{}]

    Ls = [16, 32, 16, 32, 16]
    Ms = [16, 16, 32, 32, 16]
    Ns = [16, 16, 16, 16, 32]

    for filename in filenames:
        print("********* Processing file", filename, "*********")
        eq = desc.io.load(filename)
        try:
            eq = eq[-1]
        except:
            pass

        results = []
        for grid_type, kwargs in zip(grid_types, kwargss):
            for L, M, N in zip(Ls, Ms, Ns):
                results.append(test(eq, grid_type, kwargs, L, M, N))

        results = np.array(results)
        np.testing.assert_allclose(results, np.mean(results), rtol=0.006)


def test_mercier_value():
    """
    Confirm the specific value of the Mercier objective function.
    """
    filenames = [
        ".//tests//inputs//LandremanPaul2022_QA_reactorScale_tinyPressure_lowRes.h5",
        ".//tests//inputs//high_aspect_ratio_tokamak_tinyBeta_output.h5",
        ".//tests//inputs//circular_model_tokamak_finiteBeta_output.h5",
    ]

    for filename in filenames:
        print("********* Processing file", filename, "*********")
        eq = desc.io.load(filename)
        try:
            eq = eq[-1]
        except:
            pass

        # Need to over-sample because otherwise eq.compute() switches to higher-resolution grids
        # for 0D and flux-function quantities, causing the 2 methods of
        # computing D_Mercier to give different values. See desc issue #683
        grid = QuadratureGrid(
            L=eq.L_grid,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
        )

        data = eq.compute(["D_Mercier", "G", "V", "<|B|>_rms", "p_r", "rho"], grid=grid)
        normalization = Mercier_normalization(data, eq.Psi)

        print(
            "max(D_Mercier / normalization):", np.max(data["D_Mercier"] / normalization)
        )
        print(
            "min(D_Mercier / normalization):", np.min(data["D_Mercier"] / normalization)
        )

        def test(threshold):
            integrand = (
                np.maximum(0.0, threshold - data["D_Mercier"] / normalization) ** 2
            )
            expected = 0.5 * np.sum(integrand * grid.weights) / (4 * np.pi**2)

            obj = ObjectiveFunction(
                MercierThreshold(
                    grid=grid,
                    threshold=threshold,
                    eq=eq,
                ),
            )
            obj.build()
            scalar_objective = obj.compute_scalar(obj.x(eq))
            rel_diff = abs(
                (scalar_objective - expected) / (0.5 * (scalar_objective + expected))
            )
            print(
                f"threshold: {threshold}  obj: {scalar_objective:11.9g}  expected: {expected}  rel diff: {rel_diff}"
            )
            np.testing.assert_allclose(
                scalar_objective, expected, rtol=1e-8, atol=1e-15
            )

            return scalar_objective

        thresholds = [0, 1e-5, -1e-5, 0.04, -0.04]
        for threshold in thresholds:
            test(threshold)


def test_mercier_matches_well():
    """
    Confirm the specific value of the Mercier objective function.

    Specifically, check that the normalization approximately satisfies
    DMercier / N = -(1/V) d^2 V / d s^2
    if the d^2 V / ds^2 term dominates D_Mercier.
    """
    filename = ".//tests//inputs//high_aspect_ratio_tokamak_tinyBeta_output.h5"

    eq = desc.io.load(filename)
    try:
        eq = eq[-1]
    except:
        pass

    grid = QuadratureGrid(
        L=eq.L,
        M=eq.M,
        N=eq.N,
        NFP=eq.NFP,
    )

    data = eq.compute(["rho", "V", "V_r(r)", "V_rr(r)"], grid=grid)
    d2_volume_d_s2 = (data["V_rr(r)"] - data["V_r(r)"] / data["rho"]) / (
        4 * data["rho"] ** 2
    )
    d2_volume_d_s2_over_V = d2_volume_d_s2 / data["V"]
    D_Mercier_normalized = -d2_volume_d_s2_over_V
    print("max d2_volume_d_s2:", np.max(d2_volume_d_s2))
    print("min d2_volume_d_s2:", np.min(d2_volume_d_s2))
    print("max d2_volume_d_s2_over_V:", np.max(d2_volume_d_s2_over_V))
    print("min d2_volume_d_s2_over_V:", np.min(d2_volume_d_s2_over_V))

    def test(threshold):
        integrand = np.maximum(0.0, threshold - D_Mercier_normalized) ** 2
        expected = 0.5 * np.sum(integrand * grid.weights) / (4 * np.pi**2)

        obj = ObjectiveFunction(
            MercierThreshold(
                grid=grid,
                threshold=threshold,
                well_only=True,
                eq=eq,
            ),
        )
        obj.build()
        scalar_objective = obj.compute_scalar(obj.x(eq))
        rel_diff = abs(
            (scalar_objective - expected) / (0.5 * (scalar_objective + expected))
        )
        print(
            f"threshold: {threshold}  obj: {scalar_objective:11.9g}  expected: {expected}  rel diff: {rel_diff}"
        )
        np.testing.assert_allclose(scalar_objective, expected, rtol=1e-2)

        return scalar_objective

    thresholds = [0, 3e-6, -3e-6, 2e-5, -2e-5, 4e-5]
    for threshold in thresholds:
        test(threshold)


def test_independent_of_size_and_B():
    """
    The magnetic well objective should be unchanged under scaling the size or field strength of a configuration
    """
    filenames = [
        ".//tests//inputs//circular_model_tokamak_finiteBeta_output.h5",
        ".//tests//inputs//circular_model_tokamak_finiteBeta_2xB_output.h5",
        ".//tests//inputs//circular_model_tokamak_finiteBeta_2xSize_output.h5",
    ]

    def test(eq):
        grid = QuadratureGrid(
            L=eq.L,
            M=eq.M,
            N=eq.N,
            NFP=eq.NFP,
        )
        obj = ObjectiveFunction(
            MercierThreshold(
                grid=grid,
                threshold=0.1,
                eq=eq,
            ),
        )
        obj.build()
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(f"obj: {scalar_objective:11.9g}  file: {filename}")
        assert np.abs(scalar_objective) > 0.0001
        return scalar_objective

    results = []
    for filename in filenames:
        eq = desc.io.load(filename)[-1]
        results.append(test(eq))

    results = np.array(results)
    print(results)

    # Results should all be the same:
    np.testing.assert_allclose(results, np.mean(results), rtol=0.01)
