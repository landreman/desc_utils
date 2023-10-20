import numpy as np
import pytest

import desc.io
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives import *
from desc.compute.utils import surface_averages

from desc_utils import BTarget, BContourAngle


def test_BTarget_resolution():
    """
    Confirm that the B_target objective function is
    approximately independent of grid resolution.
    """
    filenames = [
        ".//tests//inputs//HELIOTRON_MJL.h5",
        ".//tests//inputs//HELIOTRON_MJL_current.h5",
    ]

    def test(eq, grid_type, L, M, N):
        grid = grid_type(
            L=L,
            M=M,
            N=N,
            NFP=eq.NFP,
        )
        target = np.ones_like(grid.weights)
        print("target shape:", target.shape)
        obj = ObjectiveFunction(
            BTarget(
                grid=grid,
                B_target=target,
                eq=eq,
            ),
        )
        obj.build()
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
        results = []
        for grid_type in grid_types:
            for L, M, N in zip(Ls, Ms, Ns):
                results.append(test(eq, grid_type, L, M, N))

        results = np.array(results)
        # Tolerance must be loose because rho convergence is poor without sqrt(g)
        np.testing.assert_allclose(results, np.mean(results), rtol=0.005)


def test_BTarget_independent_of_size_and_B():
    """
    The BTarget objective should be unchanged under scaling the size or field strength of a configuration
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
        typical_B = eq.compute("<|B|>_rms")["<|B|>_rms"]
        obj = ObjectiveFunction(
            BTarget(
                grid=grid,
                eq=eq,
                weight=1 / typical_B,
            ),
        )
        obj.build()
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
    np.testing.assert_allclose(results, np.mean(results), rtol=2e-6)


def test_BTarget_value():
    """ """
    filename = ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5"
    print(filename)
    eq = desc.io.load(filename)

    grid = LinearGrid(
        rho=1,
        M=8,
        N=8,
        NFP=eq.NFP,
    )

    def test(target):
        obj = ObjectiveFunction(
            BTarget(
                grid=grid,
                B_target=target,
                eq=eq,
            ),
        )
        obj.build()
        scalar_objective = obj.compute_scalar(obj.x(eq))

        modB = eq.compute(["|B|"], grid=grid)["|B|"]
        expected = 0.5 / (4 * np.pi**2) * sum((modB - target) ** 2 * grid.weights)
        rel_diff = abs(
            (scalar_objective - expected) / (0.5 * (scalar_objective + expected))
        )
        print(
            f"target: {target}  obj: {scalar_objective:11.9g}  "
            f"expected: {expected}  rel diff: {rel_diff}"
        )
        np.testing.assert_allclose(scalar_objective, expected, rtol=1e-10)

        return scalar_objective

    targets = [-0.6, 0, 0.7]
    for target in targets:
        test(target)


def test_BTarget_value_axis():
    """ """
    filename = ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5"
    print(filename)
    eq = desc.io.load(filename)

    grid = LinearGrid(
        rho=0,
        theta=0,
        N=8,
        NFP=eq.NFP,
    )

    def test(target):
        B_target = 1.2 + target * np.cos(grid.nodes[:, 2])
        obj = ObjectiveFunction(
            BTarget(
                grid=grid,
                B_target=B_target,
                eq=eq,
            ),
        )
        obj.build()
        scalar_objective = obj.compute_scalar(obj.x(eq))

        modB = eq.compute(["|B|"], grid=grid)["|B|"]
        expected = 0.5 / (4 * np.pi**2) * sum((modB - B_target) ** 2 * grid.weights)
        rel_diff = abs(
            (scalar_objective - expected) / (0.5 * (scalar_objective + expected))
        )
        print(
            f"target: {target}  obj: {scalar_objective:11.9g}  "
            f"expected: {expected}  rel diff: {rel_diff}"
        )
        np.testing.assert_allclose(scalar_objective, expected, rtol=1e-10)

        return scalar_objective

    targets = [-0.6, 0, 0.7]
    for target in targets:
        test(target)


def test_BContourAngle_resolution_and_value():
    """
    Confirm that the BContourAngle objective function is
    approximately independent of grid resolution.
    """
    filenames = [
        ".//tests//inputs//LandremanPaul2022_QA_reactorScale_tinyPressure_lowRes.h5",
    ]

    def test(eq, M, N):
        grid = LinearGrid(
            rho=1,
            M=M,
            N=N,
            NFP=eq.NFP,
        )

        regularization = 0.1
        obj_term = BContourAngle(
            grid=grid,
            eq=eq,
            regularization=regularization,
        )
        obj = ObjectiveFunction(obj_term)
        obj.build()
        scalar_objective = obj.compute_scalar(obj.x(eq))

        data = eq.compute(
            [
                "iota",
                "B0",
                "B_theta",
                "B_zeta",
                "|B|_t",
                "|B|_z",
                "G",
                "I",
                "B*grad(|B|)",
                "|B|",
                "sqrt(g)",
            ],
            grid=grid,
        )

        B_cross_grad_B_dot_grad_psi = data["B0"] * (
            -data["B_zeta"] * data["|B|_t"] + data["B_theta"] * data["|B|_z"]
        )

        dB_dtheta = (data["I"] * data["B*grad(|B|)"] - B_cross_grad_B_dot_grad_psi) / (
            data["|B|"] ** 2
        )
        dB_dzeta = (
            data["G"] * data["B*grad(|B|)"] - data["iota"] * B_cross_grad_B_dot_grad_psi
        ) / (data["|B|"] ** 2)

        squared_zeta_contour_component_regularized = (dB_dtheta**2) / (
            dB_dtheta**2 + dB_dzeta**2 + regularization * data["|B|"] ** 2
        )
        should_be = 0.5 * surface_averages(
            grid,
            squared_zeta_contour_component_regularized,
            sqrt_g=data["sqrt(g)"],
            expand_out=False,
        )
        assert should_be.shape == (1,)
        should_be = should_be[0]
        print(
            f"M: {M:2}  N: {N:2}  obj: {scalar_objective:11.9g}  should be: {should_be:11.9g}"
        )
        np.testing.assert_allclose(scalar_objective, should_be)
        return scalar_objective

    # Loop over grid resolutions:
    Ms = [8, 16, 8]
    Ns = [8, 8, 16]

    for filename in filenames:
        print("********* Processing file", filename, "*********")
        eq = desc.io.load(filename)
        results = []
        for M, N in zip(Ms, Ns):
            results.append(test(eq, M, N))

        results = np.array(results)
        np.testing.assert_allclose(results, np.mean(results), rtol=1e-3)


def test_BContourAngle_independent_of_size_and_B():
    """
    The BTarget objective should be unchanged under scaling the size or field strength of a configuration
    """
    filenames = [
        ".//tests//inputs//circular_model_tokamak_output.h5",
        ".//tests//inputs//circular_model_tokamak_2xB_output.h5",
        ".//tests//inputs//circular_model_tokamak_2xSize_output.h5",
    ]

    def test(eq):
        M = eq.M
        grid = LinearGrid(
            rho=1,
            M=M,
            N=M,  # Note M instead of N here
            NFP=eq.NFP,
        )
        obj = ObjectiveFunction(
            BContourAngle(
                grid=grid,
                eq=eq,
                regularization=0.1,
            ),
        )
        obj.build()
        scalar_objective = obj.compute_scalar(obj.x(eq))
        print(f"obj: {scalar_objective:11.9g}  file: {filename}  M: {M}")
        assert np.abs(scalar_objective) > 0.01
        return scalar_objective

    results = []
    for filename in filenames:
        eq = desc.io.load(filename)[-1]
        results.append(test(eq))

    results = np.array(results)
    print(results)

    # Results should all be the same:
    np.testing.assert_allclose(results, np.mean(results), rtol=3e-5)
