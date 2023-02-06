"""Tests for bootstrap current functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import elementary_charge
from scipy.integrate import quad

import desc.io
from desc.compute.utils import compress, expand
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives import *
from desc.optimize import Optimizer
from desc.profiles import PowerSeriesProfile, SplineProfile

from desc_utils import BootstrapRedlConsistencyNormalized


class TestBootstrapObjectives:
    """Tests for bootstrap current objective functions."""

    def test_BootstrapRedlConsistencyNormalized_normalization(self):
        """Objective function should be invariant under scaling |B| or size."""
        Rmajor = 1.7
        aminor = 0.2
        Delta = 0.3
        NFP = 5
        Psi0 = 1.2
        LMN_resolution = 5
        helicity = (1, NFP)
        ne0 = 3.0e20
        Te0 = 15e3
        Ti0 = 14e3
        ne = PowerSeriesProfile(ne0 * np.array([1, -0.85]), modes=[0, 4])
        Te = PowerSeriesProfile(Te0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6])
        Ti = PowerSeriesProfile(Ti0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6])
        Zeff = 1.4

        surface = FourierRZToroidalSurface(
            R_lmn=np.array([Rmajor, aminor, Delta]),
            modes_R=[[0, 0], [1, 0], [0, 1]],
            Z_lmn=np.array([-aminor, Delta]),
            modes_Z=[[-1, 0], [0, -1]],
            NFP=NFP,
        )

        eq = Equilibrium(
            surface=surface,
            electron_density=ne,
            electron_temperature=Te,
            ion_temperature=Ti,
            atomic_number=Zeff,
            iota=PowerSeriesProfile([1.1]),
            Psi=Psi0,
            NFP=NFP,
            L=LMN_resolution,
            M=LMN_resolution,
            N=LMN_resolution,
            L_grid=2 * LMN_resolution,
            M_grid=2 * LMN_resolution,
            N_grid=2 * LMN_resolution,
            sym=True,
        )
        grid = QuadratureGrid(
            L=LMN_resolution,
            M=LMN_resolution,
            N=LMN_resolution,
            NFP=eq.NFP,
        )
        obj = ObjectiveFunction(
            BootstrapRedlConsistency(grid=grid, helicity=helicity), eq
        )
        np.set_printoptions(linewidth=400)
        scalar_objective1 = obj.compute_scalar(obj.x(eq))
        print(scalar_objective1)
        data = eq.compute(["<J*B>", "<J*B> Redl"], grid=grid)
        print("<J*B>:     ", compress(grid, data["<J*B>"]))
        print("<J*B> Redl:", compress(grid, data["<J*B> Redl"]))

        # Scale |B|, changing <J*B> and <J*B>_Redl by the following factor:
        factor = 2.0
        eq.Psi *= np.sqrt(factor)
        eq.electron_density = PowerSeriesProfile(
            factor ** (3.0 / 5) * ne0 * np.array([1, -0.85]), modes=[0, 4]
        )
        eq.electron_temperature = PowerSeriesProfile(
            factor ** (2.0 / 5) * Te0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6]
        )
        eq.ion_temperature = PowerSeriesProfile(
            factor ** (2.0 / 5) * Ti0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6]
        )
        obj = ObjectiveFunction(
            BootstrapRedlConsistency(grid=grid, helicity=helicity), eq
        )
        scalar_objective2 = obj.compute_scalar(obj.x(eq))
        print(scalar_objective2)
        data = eq.compute(["<J*B>", "<J*B> Redl"], grid=grid)
        print("<J*B>:     ", compress(grid, data["<J*B>"]))
        print("<J*B> Redl:", compress(grid, data["<J*B> Redl"]))

        # Scale size, changing <J*B> and <J*B>_Redl by the following factor:
        factor = 3.0
        eq.Psi = Psi0 / factor**2
        eq.R_lmn /= factor
        eq.Z_lmn /= factor
        eq.Rb_lmn /= factor
        eq.Zb_lmn /= factor
        eq.electron_density = PowerSeriesProfile(
            factor ** (2.0 / 5) * ne0 * np.array([1, -0.85]), modes=[0, 4]
        )
        eq.electron_temperature = PowerSeriesProfile(
            factor ** (-2.0 / 5) * Te0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6]
        )
        eq.ion_temperature = PowerSeriesProfile(
            factor ** (-2.0 / 5) * Ti0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6]
        )
        obj = ObjectiveFunction(
            BootstrapRedlConsistency(grid=grid, helicity=helicity), eq
        )
        scalar_objective3 = obj.compute_scalar(obj.x(eq))
        print(scalar_objective3)
        data = eq.compute(["<J*B>", "<J*B> Redl"], grid=grid)
        print("<J*B>:     ", compress(grid, data["<J*B>"]))
        print("<J*B> Redl:", compress(grid, data["<J*B> Redl"]))

        results = np.array([scalar_objective1, scalar_objective2, scalar_objective3])
        # Results are not perfectly identical because ln(Lambda) is not quite invariant.
        np.testing.assert_allclose(results, np.mean(results), rtol=1e-3)

    def test_BootstrapRedlConsistencyNormalized_12(self):
        """Test for bootstrap current in vacuum field.

        The BootstrapRedlConsistencyNormalized objective function should be
        1/2 if the equilibrium is a vacuum field.
        """
        helicity = (1, -4)
        filename = ".//tests//inputs//LandremanPaul2022_QH_reactorScale_lowRes.h5"
        eq = desc.io.load(filename)
        # n and T profiles do not go to 0 at rho=0 to avoid divide-by-0 there:
        eq.electron_density = PowerSeriesProfile(
            5.0e20 * np.array([1, -0.9]), modes=[0, 8]
        )
        eq.electron_temperature = PowerSeriesProfile(
            8e3 * np.array([1, -0.9]), modes=[0, 2]
        )
        eq.ion_temperature = PowerSeriesProfile(9e3 * np.array([1, -0.9]), modes=[0, 2])
        eq.atomic_number = PowerSeriesProfile(np.array([1.2, 0.2]), modes=[0, 2])

        def test(grid_type, kwargs, L_factor, M_factor, N_factor):
            grid = grid_type(
                L=eq.L * L_factor,
                M=eq.M * M_factor,
                N=eq.N * N_factor,
                NFP=eq.NFP,
                **kwargs,
            )
            obj = ObjectiveFunction(
                BootstrapRedlConsistencyNormalized(
                    grid=grid,
                    helicity=helicity,
                ),
                eq,
            )
            scalar_objective = obj.compute_scalar(obj.x(eq))
            print("Should be approximately 0.5:", scalar_objective)
            return scalar_objective

        results = []

        # Loop over grid types. For LinearGrid we need to drop the
        # point at rho=0 to avoid a divide-by-0
        grid_types = [LinearGrid, QuadratureGrid]
        kwargs = [{"axis": False}, {}]

        # Loop over grid resolutions:
        L_factors = [1, 2, 1, 1, 1]
        M_factors = [1, 1, 2, 1, 1]
        N_factors = [1, 1, 1, 2, 1]

        for grid_type, kwargs in zip(grid_types, kwargs):
            for L_factor, M_factor, N_factor in zip(
                L_factors,
                M_factors,
                N_factors,
            ):
                results.append(test(grid_type, kwargs, L_factor, M_factor, N_factor))

        np.testing.assert_allclose(np.array(results), 0.5, rtol=2e-4)

    def test_BootstrapRedlConsistencyNormalized_resolution(self):
        """Confirm that the objective function is ~independent of grid resolution."""
        helicity = (1, 0)

        filename = ".//tests//inputs//DSHAPE_current_output.h5"
        eq = desc.io.load(filename)[-1]

        eq.electron_density = PowerSeriesProfile(
            2.0e19 * np.array([1, -0.85]), modes=[0, 4]
        )
        eq.electron_temperature = PowerSeriesProfile(
            1e3 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6]
        )
        eq.ion_temperature = PowerSeriesProfile(
            1.1e3 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6]
        )
        eq.atomic_number = 1.4

        def test(grid_type, kwargs, L, M, N):
            grid = grid_type(
                L=L,
                M=M,
                N=N,
                NFP=eq.NFP,
                **kwargs,
            )
            obj = ObjectiveFunction(
                BootstrapRedlConsistencyNormalized(
                    grid=grid,
                    helicity=helicity,
                ),
                eq,
            )
            scalar_objective = obj.compute_scalar(obj.x(eq))
            print(f"grid_type:{grid_type} L:{L} M:{M} N:{N} obj:{scalar_objective}")
            return scalar_objective

        results = []

        # Loop over grid types. For LinearGrid we need to drop the
        # point at rho=0 to avoid a divide-by-0
        grid_types = [LinearGrid, QuadratureGrid]
        kwargss = [{"axis": False}, {}]

        # Loop over grid resolutions:
        Ls = [150, 300, 150, 150]
        Ms = [10, 10, 20, 10]
        Ns = [0, 0, 0, 2]

        for grid_type, kwargs in zip(grid_types, kwargss):
            for L, M, N in zip(Ls, Ms, Ns):
                results.append(test(grid_type, kwargs, L, M, N))

        results = np.array(results)
        np.testing.assert_allclose(results, np.mean(results), rtol=0.02)

    def test_bootstrap_consistency_iota(self):
        """Try optimizing for bootstrap consistency in axisymmetry, at fixed shape.

        This version of the test covers an equilibrium with an iota
        profile rather than a current profile.
        """
        ne0 = 4.0e20
        T0 = 12.0e3
        ne = PowerSeriesProfile(ne0 * np.array([1, -1]), modes=[0, 10])
        Te = PowerSeriesProfile(T0 * np.array([1, -1]), modes=[0, 2])
        Ti = Te
        Zeff = 1

        helicity = (1, 0)
        B0 = 5.0  # Desired average |B|
        LM_resolution = 8

        initial_iota = -0.61
        num_iota_points = 21
        iota = SplineProfile(np.full(num_iota_points, initial_iota))

        # Set initial condition:
        Rmajor = 6.0
        aminor = 2.0
        NFP = 1
        surface = FourierRZToroidalSurface(
            R_lmn=np.array([Rmajor, aminor]),
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=np.array([-aminor]),
            modes_Z=[[-1, 0]],
            NFP=NFP,
        )

        eq = Equilibrium(
            surface=surface,
            electron_density=ne,
            electron_temperature=Te,
            ion_temperature=Ti,
            atomic_number=Zeff,
            iota=iota,
            Psi=B0 * np.pi * (aminor**2),
            NFP=NFP,
            L=LM_resolution,
            M=LM_resolution,
            N=0,
            L_grid=2 * LM_resolution,
            M_grid=2 * LM_resolution,
            N_grid=0,
            sym=True,
        )

        eq.solve(
            verbose=3,
            ftol=1e-8,
            constraints=get_fixed_boundary_constraints(kinetic=True),
            optimizer=Optimizer("lsq-exact"),
            objective=ObjectiveFunction(objectives=ForceBalance()),
        )

        initial_output_file = "test_bootstrap_consistency_iota_initial.h5"
        print("initial_output_file:", initial_output_file)
        eq.save(initial_output_file)

        # Done establishing the initial condition. Now set up the optimization.

        constraints = (
            ForceBalance(),
            FixBoundaryR(),
            FixBoundaryZ(),
            FixElectronDensity(),
            FixElectronTemperature(),
            FixIonTemperature(),
            FixAtomicNumber(),
            FixPsi(),
        )

        # grid for bootstrap consistency objective:
        grid = LinearGrid(
            rho=np.linspace(1e-4, 1 - 1e-4, (num_iota_points - 1) * 2 + 1),
            M=eq.M * 2,
            N=eq.N * 2,
            NFP=eq.NFP,
        )
        objective = ObjectiveFunction(
            BootstrapRedlConsistencyNormalized(
                grid=grid,
                helicity=helicity,
            )
        )
        eq, _ = eq.optimize(
            verbose=3,
            objective=objective,
            constraints=constraints,
            optimizer=Optimizer("scipy-trf"),
            ftol=1e-6,
        )

        final_output_file = "test_bootstrap_consistency_iota_final.h5"
        print("final_output_file:", final_output_file)
        eq.save(final_output_file)

        scalar_objective = objective.compute_scalar(objective.x(eq))
        assert scalar_objective < 3e-5
        data = eq.compute(
            ["<J*B>", "<J*B> Redl"],
            grid=grid,
            helicity=helicity,
        )
        J_dot_B_MHD = compress(grid, data["<J*B>"])
        J_dot_B_Redl = compress(grid, data["<J*B> Redl"])

        assert np.max(J_dot_B_MHD) < 4e5
        assert np.max(J_dot_B_MHD) > 0
        assert np.min(J_dot_B_MHD) < -5.1e6
        assert np.min(J_dot_B_MHD) > -5.4e6
        np.testing.assert_allclose(J_dot_B_MHD, J_dot_B_Redl, atol=5e5)

    def test_bootstrap_consistency_current(self):
        """
        Try optimizing for bootstrap consistency in axisymmetry, at fixed shape.

        This version of the test covers the case of an equilibrium
        with a current profile rather than an iota profile.
        """
        ne0 = 4.0e20
        T0 = 12.0e3
        ne = PowerSeriesProfile(ne0 * np.array([1, -1]), modes=[0, 10])
        Te = PowerSeriesProfile(T0 * np.array([1, -1]), modes=[0, 2])
        Ti = Te

        helicity = (1, 0)
        B0 = 5.0  # Desired average |B|
        LM_resolution = 8

        current = PowerSeriesProfile([0, -1.2e7], modes=[0, 2], sym=False)

        # Set initial condition:
        Rmajor = 6.0
        aminor = 2.0
        NFP = 1
        surface = FourierRZToroidalSurface(
            R_lmn=np.array([Rmajor, aminor]),
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=np.array([-aminor]),
            modes_Z=[[-1, 0]],
            NFP=NFP,
        )

        eq = Equilibrium(
            surface=surface,
            electron_density=ne,
            electron_temperature=Te,
            ion_temperature=Ti,
            current=current,
            Psi=B0 * np.pi * (aminor**2),
            NFP=NFP,
            L=LM_resolution,
            M=LM_resolution,
            N=0,
            L_grid=2 * LM_resolution,
            M_grid=2 * LM_resolution,
            N_grid=0,
            sym=True,
        )
        current_L = 16
        eq.current.change_resolution(current_L)

        eq.solve(
            verbose=3,
            ftol=1e-8,
            constraints=get_fixed_boundary_constraints(kinetic=True, iota=False),
            optimizer=Optimizer("lsq-exact"),
            objective=ObjectiveFunction(objectives=ForceBalance()),
        )

        initial_output_file = "test_bootstrap_consistency_current_initial.h5"
        print("initial_output_file:", initial_output_file)
        eq.save(initial_output_file)

        # Done establishing the initial condition. Now set up the optimization.

        constraints = (
            ForceBalance(),
            FixBoundaryR(),
            FixBoundaryZ(),
            FixElectronDensity(),
            FixElectronTemperature(),
            FixIonTemperature(),
            FixAtomicNumber(),
            FixCurrent(indices=[0]),
            FixPsi(),
        )

        # grid for bootstrap consistency objective:
        grid = QuadratureGrid(
            L=current_L * 2,
            M=eq.M * 2,
            N=eq.N * 2,
            NFP=eq.NFP,
        )
        objective = ObjectiveFunction(
            BootstrapRedlConsistencyNormalized(
                grid=grid,
                helicity=helicity,
            )
        )
        eq, _ = eq.optimize(
            verbose=3,
            objective=objective,
            constraints=constraints,
            optimizer=Optimizer("scipy-trf"),
            ftol=1e-6,
            gtol=0,  # It is critical to set gtol=0 when optimizing current profile!
        )

        final_output_file = "test_bootstrap_consistency_current_final.h5"
        print("final_output_file:", final_output_file)
        eq.save(final_output_file)

        scalar_objective = objective.compute_scalar(objective.x(eq))
        assert scalar_objective < 3e-5
        data = eq.compute(
            ["<J*B>", "<J*B> Redl"],
            grid=grid,
            helicity=helicity,
        )
        J_dot_B_MHD = compress(grid, data["<J*B>"])
        J_dot_B_Redl = compress(grid, data["<J*B> Redl"])

        assert np.max(J_dot_B_MHD) < 4e5
        assert np.max(J_dot_B_MHD) > 0
        assert np.min(J_dot_B_MHD) < -5.1e6
        assert np.min(J_dot_B_MHD) > -5.4e6
        np.testing.assert_allclose(J_dot_B_MHD, J_dot_B_Redl, atol=5e5)
