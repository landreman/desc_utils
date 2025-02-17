"""Objectives related to rotational transform"""

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import (
    get_profiles,
    get_transforms,
)
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.utils import Timer


class MeanIota(_Objective):
    r"""Targets a radially averaged rotational transform.

    f = (1/2) [(\int d\rho iota) - target]^2

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=1``.",
        bounds_default="``target=1``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
        loss_detail=" Note: Has no effect for this objective.",
    )

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "Mean rotational transform: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="mean rotational transform",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 1
        self._grid = grid
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
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
            grid = QuadratureGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
            )
        else:
            grid = self._grid

        self._dim_f = 1
        self._data_keys = ["iota"]

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
        """Compute rotational transform profile errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium or surface degrees of freedom, eg
            Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        iota : ndarray
            rotational transform on specified flux surfaces.
        """
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=1``.",
        bounds_default="``target=1``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
        loss_detail=" Note: Has no effect for this objective.",
    )

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "Rotational transform: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="rotational transform",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 1
        self._grid = grid
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
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
                rho=0.5,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
            )
        else:
            grid = self._grid

        if grid.num_rho > 1:
            raise ValueError("For IotaAt, grid should have only a single rho value.")

        if not isinstance(grid, LinearGrid):
            raise ValueError("For IotaAt, grid should be a LinearGrid.")

        self._dim_f = 1
        self._data_keys = ["iota"]

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
        """Compute rotational transform profile errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium or surface degrees of freedom, eg
            Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        iota : ndarray
            rotational transform on specified flux surfaces.
        """
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
