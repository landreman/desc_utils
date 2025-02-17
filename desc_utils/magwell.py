from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import (
    get_profiles,
    get_transforms,
)
from desc.grid import QuadratureGrid
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.utils import Timer


class MagneticWellThreshold(_Objective):
    r"""d^2 volume / d psi^2 objective

    \int_0^1 d\rho max(0, (d^2 volume / d s^2) / V - threshold)^2

    where s = \rho^2.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    threshold: float
        Set to a negative number to provide some margin

    """
    __doc__ = __doc__.rstrip() + collect_docs(
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Magnetic well objective: "

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
        name="V''(s)",
        jac_chunk_size=None,
        threshold=0.0,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
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
            grid = QuadratureGrid(
                L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
            )
        else:
            grid = self._grid

        self._dim_f = grid.num_rho
        self._data_keys = ["V_rr(r)", "V_r(r)", "rho", "V"]

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
            Dictionary of equilibrium or surface degrees of freedom,
            eg Equilibrium.params_dict
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
        rho_weights = grid.compress(grid.spacing[:, 0])

        d2_volume_d_s2 = grid.compress(
            (data["V_rr(r)"] - data["V_r(r)"] / data["rho"]) / (4 * data["rho"] ** 2),
        )
        return jnp.maximum(0.0, d2_volume_d_s2 / data["V"] - self.threshold) * jnp.sqrt(
            rho_weights
        )
