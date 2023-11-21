"""Objectives related to the bootstrap current profile."""

import warnings

import numpy as np

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.utils import Timer

from desc.objectives.objective_funs import _Objective


class BootstrapRedlConsistencyNormalized(_Objective):
    r"""Promote consistency of the bootstrap current for axisymmetry or quasisymmetry.

    The scalar objective is defined as in eq (15) of
    Landreman, Buller, & Drevlak, Physics of Plasmas 29, 082501 (2022)
    https://doi.org/10.1063/5.0098166
    except with an integral in rho instead of s = rho**2, and with a factor of
    1/2 for consistency with desc conventions.

    f_{boot} = numerator / denominator

    where

    numerator = (1/2) \int_0^1 d\rho [<J \cdot B>_{MHD} - <J \cdot B>_{Redl}]^2

    denominator = \int_0^1 d\rho [<J \cdot B>_{MHD} + <J \cdot B>_{Redl}]^2

    <J \cdot B>_{MHD} is the parallel current profile of the MHD equilibrium, and

    <J \cdot B>_{Redl} is the parallel current profile from drift-kinetic physics

    The denominator serves as a normalization so f_{boot} is dimensionless, and
    f_{boot} = 1/2 when either <J \cdot B>_{MHD} or <J \cdot B>_{Redl} vanishes. Note
    that the scalar objective is approximately independent of grid resolution.

    The objective is treated as a sum of Nr least-squares terms, where Nr is the number
    of rho grid points. In other words, the contribution to the numerator from each rho
    grid point is returned as a separate entry in the returned vector of residuals,
    each weighted by the square root of the denominator.

    Parameters
    ----------
    helicity : 2-element tuple
        First entry must be 1. Second entry is the toroidal mode number of
        quasisymmetry, used for evaluating the Redl bootstrap current
        formula. Set to 0 for axisymmetry or quasi-axisymmetry; set to +/- NFP for
        quasi-helical symmetry.
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: Has no effect for this objective.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Note: Has no effect for this objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Bootstrap current self-consistency: {:10.3e} "

    def __init__(
        self,
        eq,
        helicity=(1, 0),
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        grid=None,
        name="Bootstrap current self-consistency (Redl)",
    ):
        if target is None and bounds is None:
            target = 0
        assert (len(helicity) == 2) and (int(helicity[1]) == helicity[1])
        assert helicity[0] == 1, "Redl bootstrap current model assumes helicity[0] == 1"

        self.helicity = helicity
        self._grid = grid
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            name=name,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if self._grid is None:
            grid = LinearGrid(
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=np.linspace(1 / 5, 1, 5),
            )
        else:
            grid = self._grid

        assert (
            self.helicity[1] == 0 or abs(self.helicity[1]) == eq.NFP
        ), "Helicity toroidal mode number should be 0 (QA) or +/- NFP (QH)"
        self._dim_f = grid.num_rho
        self._data_keys = ["<J*B>", "<J*B> Redl"]

        if eq.electron_temperature is None:
            raise RuntimeError(
                "Bootstrap current calculation requires an electron temperature "
                "profile."
            )
        if eq.electron_density is None:
            raise RuntimeError(
                "Bootstrap current calculation requires an electron density profile."
            )
        if eq.ion_temperature is None:
            raise RuntimeError(
                "Bootstrap current calculation requires an ion temperature profile."
            )

        # Try to catch cases in which density or temperatures are specified in the
        # wrong units. Densities should be ~ 10^20, temperatures are ~ 10^3.
        rho = eq.compute("rho", grid=grid)["rho"]
        if jnp.any(eq.electron_temperature(rho) > 50e3):
            warnings.warn(
                "Electron temperature is surprisingly high. It should have units of eV"
            )
        if jnp.any(eq.ion_temperature(rho) > 50e3):
            warnings.warn(
                "Ion temperature is surprisingly high. It should have units of eV"
            )
        # Profiles may go to 0 at rho=1, so exclude the last few grid points from lower
        # bounds:
        rho = rho[rho < 0.85]
        if jnp.any(eq.electron_density(rho) < 1e17):
            warnings.warn(
                "Electron density is surprisingly low. It should have units 1/meters^3"
            )
        if jnp.any(eq.electron_temperature(rho) < 30):
            warnings.warn(
                "Electron temperature is surprisingly low. It should have units of eV"
            )
        if jnp.any(eq.ion_temperature(rho) < 30):
            warnings.warn(
                "Ion temperature is surprisingly low. It should have units of eV"
            )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the bootstrap current self-consistency objective.

        Returns
        -------
        obj : ndarray
            Bootstrap current self-consistency residual on each rho grid point.

        """
        if constants is None:
            constants = self._constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            helicity=self.helicity,
        )
        grid = constants["transforms"]["grid"]
        rho_weights = grid.compress(grid.spacing[:, 0])

        denominator = jnp.sum(
            (data["<J*B>"] + data["<J*B> Redl"]) ** 2 * grid.weights
        ) / (4 * jnp.pi * jnp.pi)

        return grid.compress(
            (data["<J*B>"] - data["<J*B> Redl"]),
        ) * jnp.sqrt(rho_weights / denominator)
