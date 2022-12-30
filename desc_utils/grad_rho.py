from desc.backend import jnp
from desc.compute import (
    compute_geometry,
    compute_jacobian,
    compute_contravariant_metric_coefficients,
    get_profiles,
    get_transforms,
)
from desc.compute.utils import compress
from desc.objectives.objective_funs import _Objective
from desc.transform import Transform
from desc.utils import Timer


class GradRho(_Objective):
    """objective to control elongation.

    f = (1/2) * < max[0, a * |grad rho| - k] ** 2>
    where k is a constant threshold, and < .. > is a flux surface average.

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
        k in the formula above

    """

    _scalar = False
    _linear = False
    _units = "(dimensionless)"
    print_value_fmt = "|grad rho| penalty: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        normalize=False,
        normalize_target=False,
        vol_grid=None,
        surf_grid=None,
        name="|grad rho|",
        threshold=0.0,
    ):

        self.surf_grid = surf_grid
        self.vol_grid = vol_grid
        self.threshold = threshold
        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
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
        if self.surf_grid is None:
            self.surf_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, rho=[1.0], NFP=eq.NFP)
        if self.vol_grid is None:
            self.vol_grid = QuadratureGrid(
                L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
            )

        self._dim_f = self.surf_grid.num_nodes
        self._surf_data_keys = ["sqrt(g)", "|grad(rho)|"]
        self._vol_data_keys = ["a"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._surf_profiles = get_profiles(
            *self._surf_data_keys, eq=eq, grid=self.surf_grid
        )
        self._surf_transforms = get_transforms(
            *self._surf_data_keys, eq=eq, grid=self.surf_grid
        )
        self._vol_profiles = get_profiles(
            *self._vol_data_keys, eq=eq, grid=self.vol_grid
        )
        self._vol_transforms = get_transforms(
            *self._vol_data_keys, eq=eq, grid=self.vol_grid
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, **kwargs):
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
        params = {
            "R_lmn": R_lmn,
            "Z_lmn": Z_lmn,
        }
        surf_data = compute_jacobian(
            params,
            self._surf_transforms,
            self._surf_profiles,
        )
        surf_data = compute_contravariant_metric_coefficients(
            params, self._surf_transforms, self._surf_profiles, data=surf_data
        )
        vol_data = compute_geometry(
            params,
            self._vol_transforms,
            self._vol_profiles,
        )

        Vprime = jnp.sum(self.surf_grid.weights * surf_data["sqrt(g)"])
        residuals = jnp.sqrt(
            self.surf_grid.weights * surf_data["sqrt(g)"] / Vprime
        ) * jnp.maximum(0.0, vol_data["a"] * surf_data["|grad(rho)|"] - self.threshold)

        return self._shift_scale(residuals)
