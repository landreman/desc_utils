from scipy.constants import mu_0

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import (
    get_profiles,
    get_transforms,
)
from desc.grid import QuadratureGrid
from desc.objectives.objective_funs import _Objective
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

    _units = "(dimensionless)"
    _print_value_fmt = "Mercier threshold objective: {:10.3e} "

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
        name="Mercier",
        threshold=0.0,
        well_only=False,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.threshold = threshold
        if well_only:
            self.Mercier_term = "D_well"
        else:
            self.Mercier_term = "D_Mercier"

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
            grid = QuadratureGrid(
                L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
            )
            print("build: using new grid")
        else:
            print("build: using supplied grid")
            grid = self._grid

        self._dim_f = grid.num_rho
        self._data_keys = [self.Mercier_term, "G", "V", "<|B|>_rms", "p_r", "rho"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        print("data keys:", self._data_keys)
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the objective"""
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

        normalization = Mercier_normalization(data, params["Psi"])
        DMercier_normalized = grid.compress(
            data[self.Mercier_term] / normalization
        )

        return jnp.maximum(0.0, self.threshold - DMercier_normalized) * jnp.sqrt(
            rho_weights
        )
