from scipy.constants import mu_0

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


def Mercier_normalization(data, Psi):
    """
    Return the rho-dependent quantity used to normalize D_Mercier.

    See ``20230124-01 Normalizations for Mercier objective.lyx`` for the derivation.

    Parameters
    ----------
    data : dict
        data returned by e.g. ``eq.compute()``.
        Keys must include ``V``, ``G``, ``p_r``, ``rho``, and ``<|B|>_rms``.
    Psi : float
        Total toroidal flux at the boundary.
    """
    normalization = jnp.abs(
        mu_0
        * data["V"]
        * data["G"]
        * data["p_r"]
        / (4 * Psi**4 * data["rho"] ** 3 * data["<|B|>_rms"])
    )
    return normalization


class MercierThreshold(_Objective):
    r"""Mercier stability objective

    \int_0^1 d\rho max(0, threshold - DMercier/N)^2

    where N is a positive normalization.

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
    well_only: Boolean
        Set to ``True`` to use ``D_well`` instead of the total ``D_Mercier``. Used for testing.

    """

    _scalar = True
    _linear = False
    _units = "(dimensionless)"
    _print_value_fmt = "Mercier threshold objective: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="Mercier",
        threshold=0.0,
        well_only=False,
    ):

        self.grid = grid
        self.threshold = threshold
        if well_only:
            self.Mercier_term = "D_well"
        else:
            self.Mercier_term = "D_Mercier"

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
                L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
            )

        self._dim_f = self.grid.num_rho
        self._data_keys = [self.Mercier_term, "G", "V", "<|B|>_rms", "p_r", "rho"]
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
        """Compute the objective"""
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )
        rho_weights = compress(self.grid, self.grid.spacing[:, 0])

        normalization = Mercier_normalization(data, params["Psi"])
        DMercier_normalized = compress(
            self.grid, data[self.Mercier_term] / normalization
        )

        residuals = jnp.maximum(0.0, self.threshold - DMercier_normalized) * jnp.sqrt(
            rho_weights
        )

        return self._shift_scale(residuals)
