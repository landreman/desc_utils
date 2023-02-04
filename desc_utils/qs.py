from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import (
    get_params,
    get_profiles,
    get_transforms,
)
from desc.compute.utils import compress
from desc.objectives.objective_funs import _Objective
from desc.transform import Transform
from desc.utils import Timer


class QuasisymmetryTwoTermNormalized(_Objective):
    """Quasi-symmetry two-term error.

    Differences from QuasisymmetryTwoTerm:
    * f_C is divided by B^3 so the objective is unchanged by scaling B.
    * scaling is changed so the objective is approximately independent of grid resolution.

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
    helicity : tuple, optional
        Type of quasi-symmetry (M, N).
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _units = "(dimensionless)"
    _print_value_fmt = "Quasisymmetry error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        helicity=(1, 0),
        name="QS two-term normalized",
    ):

        self.grid = grid
        self._helicity = helicity
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
        if self.grid is None:
            self.grid = QuadratureGrid(
                L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
            )

        self._dim_f = self.grid.num_nodes
        self._data_keys = ["f_C", "|B|", "sqrt(g)"]
        self._args = get_params(self._data_keys)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=self.grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=self.grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute quasi-symmetry two-term errors.

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
        f : ndarray
            Quasi-symmetry flux function error at each node.

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
            helicity=self._helicity,
        )

        denominator = jnp.sum(self.grid.weights * jnp.abs(data["sqrt(g)"]))
        return (
            data["f_C"]
            / (data["|B|"] ** 3)
            * jnp.sqrt(self.grid.weights * jnp.abs(data["sqrt(g)"]) / denominator)
        )

    @property
    def helicity(self):
        """tuple: Type of quasi-symmetry (M, N)."""
        return self._helicity
