from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import (
    get_params,
    get_profiles,
    get_transforms,
)
from desc.grid import QuadratureGrid
from desc.objectives.objective_funs import _Objective
from desc.utils import Timer


class MagneticWellThreshold(_Objective):
    r"""d^2 volume / d psi^2 objective

    \int_0^1 d\rho max(0, (d^2 volume / d s^2) / V - threshold)^2

    where s = \rho^2.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.
    threshold: float
        Set to a negative number to provide some margin

    """

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "Magnetic well objective: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="V''(s)",
        threshold=0.0,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.threshold = threshold
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
                L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
            )
        else:
            grid = self._grid

        self._dim_f = grid.num_rho
        self._data_keys = ["V_rr(r)", "V_r(r)", "rho", "V"]
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
        """Compute objective

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        V : float

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
        rho_weights = grid.compress(grid.spacing[:, 0])

        d2_volume_d_s2 = grid.compress(
            (data["V_rr(r)"] - data["V_r(r)"] / data["rho"]) / (4 * data["rho"] ** 2),
        )
        return jnp.maximum(0.0, d2_volume_d_s2 / data["V"] - self.threshold) * jnp.sqrt(
            rho_weights
        )
