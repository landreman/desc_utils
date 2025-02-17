from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import (
    get_profiles,
    get_transforms,
)
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.utils import Timer


class AxisymmetryBarrier(_Objective):
    r"""Prevents a configuration from becoming axisymmetric



    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    R_threshold: float
    Z_threshold: float

    """
    __doc__ = __doc__.rstrip() + collect_docs(
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Axisymmetry barrier: "

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
        name="Axisymmetry barrier",
        jac_chunk_size=None,
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
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
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
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

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
