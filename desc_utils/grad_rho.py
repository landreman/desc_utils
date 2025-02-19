from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import (
    get_profiles,
    get_transforms,
)
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.utils import Timer


class GradRho0(_Objective):
    """objective to control elongation.

    f = (1/2) * < max[0, a * |grad rho| - k] ** 2>
    where k is a constant threshold, < .. > is a flux surface average,
    and a is the actual minor radius.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    vol_grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    surf_grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    threshold: float
        k in the formula above

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``."
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "|grad rho| penalty: "

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
        vol_grid=None,
        surf_grid=None,
        name="|grad rho|",
        jac_chunk_size=None,
        threshold=0.0,
    ):
        if target is None and bounds is None:
            target = 0
        self._surf_grid = surf_grid
        self._vol_grid = vol_grid
        self.threshold = threshold
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
        if self._surf_grid is None:
            surf_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, rho=[1.0], NFP=eq.NFP)
        else:
            surf_grid = self._surf_grid

        if self._vol_grid is None:
            vol_grid = QuadratureGrid(
                L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
            )
        else:
            vol_grid = self._vol_grid

        self._dim_f = surf_grid.num_nodes
        self._surf_data_keys = ["sqrt(g)", "|grad(rho)|"]
        self._vol_data_keys = ["a"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        surf_profiles = get_profiles(
            self._surf_data_keys, obj=eq, grid=surf_grid
        )
        surf_transforms = get_transforms(
            self._surf_data_keys, obj=eq, grid=surf_grid
        )
        vol_profiles = get_profiles(
            self._vol_data_keys, obj=eq, grid=vol_grid
        )
        vol_transforms = get_transforms(
            self._vol_data_keys, obj=eq, grid=vol_grid
        )
        self._constants = {
            "vol_transforms": vol_transforms,
            "vol_profiles": vol_profiles,
            "surf_transforms": surf_transforms,
            "surf_profiles": surf_profiles,
            "transforms": surf_transforms,  # Desc expects this key, even though I don't use it.
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute objective

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        V : float

        """
        if constants is None:
            constants = self._constants
        vol_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._vol_data_keys,
            params=params,
            transforms=constants["vol_transforms"],
            profiles=constants["vol_profiles"],
        )
        surf_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._surf_data_keys,
            params=params,
            transforms=constants["surf_transforms"],
            profiles=constants["surf_profiles"],
        )
        vol_grid = constants["vol_transforms"]["grid"]
        surf_grid = constants["surf_transforms"]["grid"]

        Vprime = jnp.sum(surf_grid.weights * surf_data["sqrt(g)"])
        return jnp.sqrt(
            surf_grid.weights * surf_data["sqrt(g)"] / Vprime
        ) * jnp.maximum(0.0, vol_data["a"] * surf_data["|grad(rho)|"] - self.threshold)


class GradRho(_Objective):
    """objective to control elongation.

    f = (1/2) * < max[0, a * |grad rho| - k] ** 2>
    where k is a constant threshold, < .. > is a flux surface average,
    and a is target minor radius.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    a_minor : float
        Reference value of the minor radius, typically the target value.
    threshold: float
        k in the formula above

    """
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "|grad rho| penalty: "

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
        name="|grad rho|",
        jac_chunk_size=None,
        a_minor=None,
        threshold=0.0,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.a_minor = a_minor
        self.threshold = threshold
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
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, rho=[1.0], NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["sqrt(g)", "|grad(rho)|"]

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
        """Compute objective

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
        V : float

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

        Vprime = jnp.sum(grid.weights * data["sqrt(g)"])
        return jnp.sqrt(grid.weights * data["sqrt(g)"] / Vprime) * jnp.maximum(
            0.0, self.a_minor * data["|grad(rho)|"] - self.threshold
        )
