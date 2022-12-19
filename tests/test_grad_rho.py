"""Tests for |grad rho| penalty."""

import numpy as np
import pytest

import desc.io
from desc.grid import LinearGrid, QuadratureGrid
from desc.optimize import Optimizer
from desc.objectives import *

from desc_utils import GradRho


def test_grad_rho_resolution():
    """
    Confirm that the magnetic well objective function is
    approximately independent of grid resolution.
    """
    filenames = [
        ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5",
        ".//tests//inputs//HELIOTRON_MJL.h5",
    ]

    def test(eq, L, M, N):
        vol_grid = QuadratureGrid(
            L=L,
            M=M,
            N=N,
            NFP=eq.NFP,
        )
        surf_grid = LinearGrid(
            rho=1,
            M=M,
            N=N,
            NFP=eq.NFP,
        )
        obj = ObjectiveFunction(
            GradRho(
                vol_grid=vol_grid,
                surf_grid=surf_grid,
                threshold=1.1,
            ),
            eq,
        )
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(f"obj: {scalar_objective:11.9g}  L: {L}  M: {M}  N: {N}")
        return scalar_objective

    # Loop over grid resolutions:
    Ls = [8, 16, 8, 16, 8]
    Ms = [8, 8, 16, 16, 8]
    Ns = [8, 8, 8, 8, 16]

    # Ls = [16, 32, 16, 32, 16]
    # Ms = [16, 16, 32, 32, 16]
    # Ns = [16, 16, 16, 16, 32]

    for filename in filenames:
        print("********* Processing file", filename, "*********")
        eq = desc.io.load(filename)
        results = []
        for L, M, N in zip(Ls, Ms, Ns):
            results.append(test(eq, L, M, N))

        results = np.array(results)
        np.testing.assert_allclose(results, np.mean(results), rtol=1e-3)


def test_grad_rho_value():
    """
    For circular concentric surfaces, with threshold 0, the grad rho objective should be marginal.
    """
    filename = ".//tests//inputs//high_aspect_ratio_tokamak_output.h5"

    eq = desc.io.load(filename)[-1]

    vol_grid = QuadratureGrid(
        L=8,
        M=8,
        N=8,
        NFP=eq.NFP,
    )
    surf_grid = LinearGrid(
        rho=1,
        M=8,
        N=8,
        NFP=eq.NFP,
    )

    def test(threshold):
        obj = ObjectiveFunction(
            GradRho(
                vol_grid=vol_grid,
                surf_grid=surf_grid,
                threshold=threshold,
            ),
            eq,
        )
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(f"obj: {scalar_objective:11.9g}  threshold: {threshold}")

        if threshold > 1:
            assert scalar_objective < 1e-15
        else:
            expected = 0.5 * (1 - threshold) ** 2
            rel_diff = abs(
                (scalar_objective - expected) / (0.5 * (scalar_objective + threshold))
            )
            print(f"  expected: {expected}  rel diff: {rel_diff}")
            np.testing.assert_allclose(scalar_objective, expected, rtol=1e-4)

        return scalar_objective

    thresholds = [-1.1, 0, 0.5, 1.3, 5.1]
    for threshold in thresholds:
        test(threshold)
