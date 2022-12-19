"""Objectives related to rotational transform"""

from desc.backend import jnp
from desc.compute import (
    compute_rotational_transform,
    data_index,
)
from desc.compute.utils import compress
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives.objective_funs import _Objective
from desc.transform import Transform
from desc.utils import Timer


class MeanIota(_Objective):
    r"""Targets a radially averaged rotational transform.

    f = (1/2) [(\int d\rho iota) - target]^2

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f == grid.num_rho
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f == grid.num_rho
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect for this objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.
    """

    _scalar = True
    _linear = False
    _units = "(dimensionless)"
    _print_value_fmt = "Mean rotational transform: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="mean rotational transform",
    ):

        self.grid = grid
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
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
            )

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        if eq.iota is not None:
            self._iota = eq.iota.copy()
            self._iota.grid = self.grid
            self._current = None
        else:
            self._current = eq.current.copy()
            self._current.grid = self.grid
            self._iota = None

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["iota"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["iota"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["iota"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, c_l, Psi, **kwargs):
        """Compute rotational transform profile errors.

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
        iota : ndarray
            rotational transform on specified flux surfaces.
        """
        data = compute_rotational_transform(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._current,
        )
        mean_iota = jnp.sum(
            compress(
                self.grid, data["iota"] * self.grid.spacing[:, 0], surface_label="rho"
            )
        )
        return self._shift_scale(mean_iota)
