from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import (
    get_params,
    get_profiles,
    get_transforms,
)
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives.objective_funs import _Objective

# from desc.objectives import ObjectiveFromUser
from desc.utils import Timer

# def B_target(grid=None, target=None, eq=None):
#     """Return an objective to make field strength equal a target.

#     f = (1 / <B²>) ∫dρ ∫dθ ∫dζ (B - B_target)²

#     where <B²> is the volume average of B²
#     """
#     def func(grid, data):
#         print("Size of |B|:", data["|B|"].shape)
#         print("Size of target:", target.shape)
#         print("Size of weights:", grid.weights.shape)
#         print("Size of <|B|>_rms:", data["<|B|>_rms"].shape)
#         return (data["|B|"] - target) * jnp.sqrt(grid.weights / (4 * jnp.pi)) / data["<|B|>_rms"]

#     return ObjectiveFromUser(
#         func,
#         eq=eq,
#         grid=grid,
#     )


class BTarget(_Objective):
    """Return an objective to make field strength equal a target.

    f = (1 / (8π²) ∫dρ ∫dθ ∫dζ (B - B_target)²

    You should use the B_target argument, not the target argument!!

    To make this objective independent of field strength, you should also set
    weight = 1 / B0 where B0 is an expected magnitude of B, such as
    np.mean(B_target).

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. len(target) must be equal to
        Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this
        should also be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    a_minor : float
        Reference value of the minor radius, typically the target value.
    name : str
        Name of the objective function.
    threshold: float
        k in the formula above

    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "B_target: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        a_minor=None,
        name="BTarget",
        B_target=0.0,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.B_target = B_target
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
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, rho=[1.0], NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["|B|"]
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

        return (data["|B|"] - self.B_target) * jnp.sqrt(
            grid.weights / (4 * jnp.pi**2)
        )


class BContourAngle(_Objective):
    """

    f = ½ ⟨(∂B/∂θ)² / [regularization * |B|² + (∂B/∂θ)² + dBdzeta_denom_fac * (∂B/∂φ)²]⟩

    where ⟨ ⟩ is a flux surface average, and θ and φ are Boozer angles.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. len(target) must be equal to
        Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this
        should also be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "B_target: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="BContourAngle",
        regularization=0.001,
        dBdzeta_denom_fac=1.0,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.regularization = regularization
        self.dBdzeta_denom_fac = dBdzeta_denom_fac
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
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, rho=[1.0], NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = [
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
        ]
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

        B2 = data["|B|"] ** 2

        B_cross_grad_modB_dot_grad_psi = data["B0"] * (
            data["B_theta"] * data["|B|_z"] - data["B_zeta"] * data["|B|_t"]
        )

        dB_dtheta = (
            data["I"] * data["B*grad(|B|)"] - B_cross_grad_modB_dot_grad_psi
        ) / B2

        dB_dzeta = (
            data["G"] * data["B*grad(|B|)"]
            - data["iota"] * B_cross_grad_modB_dot_grad_psi
        ) / B2

        denominator = (
            dB_dtheta**2
            + self.dBdzeta_denom_fac * dB_dzeta**2
            + self.regularization * B2
        )

        Vprime = jnp.sum(grid.weights * data["sqrt(g)"])

        return dB_dtheta * jnp.sqrt(
            data["sqrt(g)"] * grid.weights / (Vprime * denominator)
        )
