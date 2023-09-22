"""Objectives related to rotational transform"""

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import (
    get_params,
    get_profiles,
    get_transforms,
)
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives.objective_funs import _Objective
from desc.utils import Timer


class MeanIota(_Objective):
    r"""Targets a radially averaged rotational transform.

    f = (1/2) [(\int d\rho iota) - target]^2

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f == grid.num_rho
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f == grid.num_rho
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect for this objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.
    """

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "Mean rotational transform: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="mean rotational transform",
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq=None, use_jit=True, verbose=1):
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
        eq = eq or self._eq
        if self._grid is None:
            grid = QuadratureGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
            )
        else:
            grid = self._grid

        self._dim_f = 1
        self._data_keys = ["iota"]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=grid.axis.size,
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

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute rotational transform profile errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        iota : ndarray
            rotational transform on specified flux surfaces.
        """
        params, constants = self._parse_args(*args, **kwargs)
        if constants is None:
            constants = self._constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        grid = constants["transforms"]["grid"]
        return jnp.sum(
            grid.compress(
                data["iota"] * grid.spacing[:, 0], surface_label="rho"
            )
        )


class IotaAt(_Objective):
    r"""Targets the rotational transform on one surface.

    f = (1/2) [iota - target]^2

    Use this objective with a LinearGrid containing only a single rho value.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f == grid.num_rho
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f == grid.num_rho
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect for this objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.
    """

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "Rotational transform: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="rotational transform",
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq=None, use_jit=True, verbose=1):
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
        eq = eq or self._eq
        if self._grid is None:
            grid = LinearGrid(
                rho=0.5,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
            )
        else:
            grid = self._grid

        if grid.num_rho > 1:
            raise ValueError("For IotaAt, grid should have only a single rho value.")

        if not isinstance(grid, LinearGrid):
            raise ValueError("For IotaAt, grid should be a LinearGrid.")

        self._dim_f = 1
        self._data_keys = ["iota"]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=grid.axis.size,
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

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute rotational transform profile errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        iota : ndarray
            rotational transform on specified flux surfaces.
        """
        params, constants = self._parse_args(*args, **kwargs)
        if constants is None:
            constants = self._constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return data["iota"][0]
