from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import (
    get_profiles,
    get_transforms,
)
from desc.compute.utils import surface_averages
from desc.grid import LinearGrid
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
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
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
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, rho=[1.0], NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["|B|"]

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
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

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

        return (data["|B|"] - self.B_target) * jnp.sqrt(
            grid.weights / (4 * jnp.pi**2)
        )


class BContourAngle(_Objective):
    """

    f = ½ ⟨(∂B/∂θ)² / [regularization * |B|² + (∂B/∂θ)² + dBdzeta_denom_fac * (∂B/∂ζ)²]⟩

    where ⟨ ⟩ is a flux surface average, and θ and ζ are Boozer angles.

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
    _print_value_fmt = "BContourAngle: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
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
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

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


class dBdThetaHeuristic(_Objective):
    """

    f = ½ ⟨w (∂B/∂θ / B)²⟩

    where ⟨ ⟩ is a flux surface average, θ and ζ are Boozer angles, and
    w is a dimensionless weight that emphasizes regions in which |∂B/∂ζ| is small.
    The exact defintion is

    w = 1 / [1 + sharpness * (∂B/∂ζ)² / ⟨(∂B/∂ζ)²⟩]

    or

    w = exp(-sharpness * (∂B/∂ζ)² / ⟨(∂B/∂ζ)²⟩)

    or

    w = exp(-sharpness * (∂B/∂ζ)⁴ / ⟨(∂B/∂ζ)⁴⟩)

    Larger values of sharpness cause the weight to be focused more on regions in
    which |∂B/∂ζ| is small.


    Parameters
    ----------
    sharpness : float
        Sharpness.
    weight_func : str
        "rational", "exp", or "exp2"

    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "dBdThetaHeuristic: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        grid=None,
        name="dBdThetaHeuristic",
        sharpness=1.0,
        weight_func="exp",
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.sharpness = sharpness

        def weight_rational(sharpness, dB_dzeta_squared_normalized):
            return 1 / (1 + sharpness * dB_dzeta_squared_normalized)

        def weight_exp(sharpness, dB_dzeta_squared_normalized):
            return jnp.exp(-sharpness * dB_dzeta_squared_normalized)

        def weight_exp2(sharpness, dB_dzeta_squared_normalized):
            return jnp.exp(-sharpness * dB_dzeta_squared_normalized**2)

        if weight_func == "rational":
            self.weight_function = weight_rational
        elif weight_func == "exp":
            self.weight_function = weight_exp
        elif weight_func == "exp2":
            self.weight_function = weight_exp2
        else:
            raise RuntimeError("Invalid setting for 'function'")

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
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

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

        dB_dzeta_squared = dB_dzeta**2
        dB_dzeta_avg_squared = surface_averages(
            grid, dB_dzeta_squared, sqrt_g=data["sqrt(g)"]
        )
        weight = self.weight_function(
            self.sharpness, dB_dzeta_squared / dB_dzeta_avg_squared
        )

        Vprime = jnp.sum(grid.weights * data["sqrt(g)"])

        return (
            2
            * jnp.pi
            * dB_dtheta
            / data["|B|"]
            * jnp.sqrt(weight * data["sqrt(g)"] * grid.weights / Vprime)
        )


class BMaxMinHeuristic(_Objective):
    """

    Set weight = 1 / B_typical

    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "BMaxMinHeuristic: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        grid=None,
        name="BMaxMinHeuristic",
    ):
        if target is None and bounds is None:
            target = 0
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
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, rho=[1.0], NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = 2
        self._data_keys = [
            "|B|",
        ]

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
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

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

        B2D = data["|B|"].reshape((grid.num_theta, grid.num_zeta), order="F")

        max_over_zeta = jnp.max(B2D, axis=1)
        min_over_zeta = jnp.min(B2D, axis=1)

        return jnp.array(
            [
                jnp.max(max_over_zeta) - jnp.min(max_over_zeta),
                jnp.max(min_over_zeta) - jnp.min(min_over_zeta),
            ]
        )


class GradB(_Objective):
    """Penalty on short field scale length

    f = (1/2) * < max[0, (a / B₀) * |grad B| - k] ** 2>
    where k is a constant threshold, < .. > is a flux surface average,
    a is target minor radius, and B₀ is a target field strength.

    Parameters
    ----------
    a_minor : float
        a in the formula above.
    B_target : float
        B₀ in the formula above.
    threshold: float
        k in the formula above.

    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "|grad B| penalty: {:10.3e} "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        grid=None,
        a_minor=None,
        B_target=None,
        name="|grad B|",
        threshold=0.0,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.a_minor = a_minor
        self.B_target = B_target
        self.threshold = threshold
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
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, rho=[1.0], NFP=eq.NFP)
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["sqrt(g)", "|grad(B)|"]

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
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

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
            0.0, (self.a_minor / self.B_target) * data["|grad(B)|"] - self.threshold
        )
