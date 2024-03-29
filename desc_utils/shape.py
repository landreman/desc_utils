from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import (
    get_profiles,
    get_transforms,
)
from desc.objectives.objective_funs import _Objective
from desc.utils import Timer


class AxisymmetryBarrier(_Objective):
    r"""Prevents a configuration from becoming axisymmetric



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
    R_threshold: float
    Z_threshold: float

    """

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "Axisymmetry barrier: {:10.3e} "

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
        name="Axisymmetry barrier",
        R_threshold=0.0,
        Z_threshold=0.0,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        assert R_threshold >= 0
        assert Z_threshold >= 0
        self.R_threshold = R_threshold
        self.Z_threshold = Z_threshold
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
        # if self.grid is None:
        #    self.grid = QuadratureGrid(
        #        L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
        #    )

        self._dim_f = 2
        self._data_keys = ["R", "Z"]

        index = None
        modes = eq.R_basis.modes
        for j in range(len(eq.R_lmn)):
            if modes[j, 0] == 0 and modes[j, 1] == 0 and modes[j, 2] == 1:
                index = j
        if index is None:
            raise RuntimeError("Did not find an R mode with m=0 and n=1")
        self.R_index = index

        index = None
        modes = eq.Z_basis.modes
        for j in range(len(eq.Z_lmn)):
            if modes[j, 0] == 0 and modes[j, 1] == 0 and modes[j, 2] == -1:
                index = j
        if index is None:
            raise RuntimeError("Did not find a Z mode with m=0 and n=-1")
        self.Z_index = index
        print("R for R_index:", eq.R_lmn[self.R_index])
        print("Z for Z_index:", eq.Z_lmn[self.Z_index])

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        # profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        # transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        #self._constants = {
        #    "transforms": transforms,
        #    "profiles": profiles,
        #}

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
        R001 = params["R_lmn"][self.R_index]
        Z001 = params["Z_lmn"][self.Z_index]
        return jnp.array(
            [
                jnp.minimum(0.0, jnp.abs(R001) - self.R_threshold),
                jnp.minimum(0.0, jnp.abs(Z001) - self.Z_threshold),
            ],
        )
