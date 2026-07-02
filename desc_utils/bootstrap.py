"""Objectives related to the bootstrap current profile."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.utils import Timer, errorif, warnif


class BootstrapRedlConsistencyNormalized(_Objective):
    r"""Promote consistency of the bootstrap current for axisymmetry or quasisymmetry.

    The scalar objective is defined as in eq (15) of
    Landreman, Buller, & Drevlak, Physics of Plasmas 29, 082501 (2022)
    https://doi.org/10.1063/5.0098166
    except with an integral in rho instead of s = rho**2, and with a factor of
    1/2 for consistency with desc conventions.

    f_{boot} = numerator / denominator

    where

    numerator = (1/2) \int_0^1 d\rho [<J \cdot B>_{MHD} - <J \cdot B>_{Redl}]^2

    denominator = \int_0^1 d\rho [<J \cdot B>_{MHD} + <J \cdot B>_{Redl}]^2

    <J \cdot B>_{MHD} is the parallel current profile of the MHD equilibrium, and

    <J \cdot B>_{Redl} is the parallel current profile from drift-kinetic physics

    The denominator serves as a normalization so f_{boot} is dimensionless, and
    f_{boot} = 1/2 when either <J \cdot B>_{MHD} or <J \cdot B>_{Redl} vanishes. Note
    that the scalar objective is approximately independent of grid resolution.

    The objective is treated as a sum of Nr least-squares terms, where Nr is the number
    of rho grid points. In other words, the contribution to the numerator from each rho
    grid point is returned as a separate entry in the returned vector of residuals,
    each weighted by the square root of the denominator.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    helicity : tuple, optional
        Type of quasi-symmetry (M, N). Default = quasi-axisymmetry (1, 0).
        First entry must be M=1. Second entry is the toroidal mode number N,
        used for evaluating the Redl bootstrap current formula. Set to 0 for
        axisymmetry or quasi-axisymmetry; set to +/-NFP for quasi-helical symmetry.
    degree : int, optional
        The `degree` kwarg to pass to the `<J*B>_Redl` compute call, which
        controls the degree of polynomial fit to the Redl current derivative
        before it is integrated.

    """
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Bootstrap current self-consistency: "

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
        helicity=(1, 0),
        degree=None,
        name="Bootstrap current self-consistency (Redl)",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        assert (len(helicity) == 2) and (int(helicity[1]) == helicity[1])
        assert helicity[0] == 1, "Redl bootstrap current model assumes helicity[0] == 1"

        self._grid = grid
        self.helicity = helicity
        self._degree = degree
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
        if self._grid is None:
            grid = LinearGrid(
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=np.linspace(1 / 5, 1, 5),
            )
        else:
            grid = self._grid

        warnif(
            (grid.num_theta * (1 + eq.sym)) < 2 * eq.M,
            RuntimeWarning,
            "BootstrapRedlConsistencyNormalized objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            RuntimeWarning,
            "BootstrapRedlConsistencyNormalized objective grid requires toroidal "
            "resolution for surface averages",
        )

        errorif(
            not (self.helicity[1] == 0 or abs(self.helicity[1]) == eq.NFP),
            ValueError,
            "Helicity toroidal mode number should be 0 (QA) or +/- NFP (QH)",
        )
        errorif(
            grid.axis.size,
            ValueError,
            "Redl formula is undefined at rho=0, but grid has grid points at rho=0",
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["<J*B>", "<J*B> Redl"]

        errorif(
            eq.electron_temperature is None,
            RuntimeError,
            "Bootstrap current calculation requires an electron temperature "
            "profile.",
        )
        errorif(
            eq.electron_density is None,
            RuntimeError,
            "Bootstrap current calculation requires an electron density profile.",
        )
        errorif(
            eq.ion_temperature is None,
            RuntimeError,
            "Bootstrap current calculation requires an ion temperature profile.",
        )

        rho = grid.compress(grid.nodes[:, 0], "rho")

        errorif(
            np.any(np.isclose(eq.electron_density(rho), 0.0, atol=1e-8)),
            ValueError,
            "Redl formula is undefined where kinetic profiles vanish, "
            "but given electron density vanishes at at least one provided"
            "rho grid point.",
        )
        errorif(
            np.any(np.isclose(eq.electron_temperature(rho), 0.0, atol=1e-8)),
            ValueError,
            "Redl formula is undefined where kinetic profiles vanish, "
            "but given electron temperature vanishes at at least one provided"
            "rho grid point.",
        )
        errorif(
            np.any(np.isclose(eq.ion_temperature(rho), 0.0, atol=1e-8)),
            ValueError,
            "Redl formula is undefined where kinetic profiles vanish, "
            "but given ion temperature vanishes at at least one provided"
            "rho grid point.",
        )
        warnif(
            jnp.any(eq.electron_density(rho) > 1e22),
            UserWarning,
            "Electron density is surprisingly high. It should have units of "
            "1/meters^3",
        )
        warnif(
            jnp.any(eq.electron_temperature(rho) > 50e3),
            UserWarning,
            "Electron temperature is surprisingly high. It should have units of eV",
        )
        warnif(
            jnp.any(eq.ion_temperature(rho) > 50e3),
            UserWarning,
            "Ion temperature is surprisingly high. It should have units of eV",
        )

        rho = rho[rho < 0.85]
        warnif(
            jnp.any(eq.electron_density(rho) < 1e17),
            UserWarning,
            "Electron density is surprisingly low. It should have units 1/meters^3",
        )
        warnif(
            jnp.any(eq.electron_temperature(rho) < 30),
            UserWarning,
            "Electron temperature is surprisingly low. It should have units of eV",
        )
        warnif(
            jnp.any(eq.ion_temperature(rho) < 30),
            UserWarning,
            "Ion temperature is surprisingly low. It should have units of eV",
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
            "helicity": self._helicity,
            "degree": self._degree,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the bootstrap current self-consistency objective.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        obj : ndarray
            Bootstrap current self-consistency residual on each rho grid point.

        """
        constants = self._get_deprecated_constants(constants)
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            helicity=constants["helicity"],
            degree=constants["degree"],
        )
        grid = constants["transforms"]["grid"]
        rho_weights = grid.compress(grid.spacing[:, 0])

        denominator = jnp.sum(
            (data["<J*B>"] + data["<J*B> Redl"]) ** 2 * grid.weights
        ) / (4 * jnp.pi * jnp.pi)

        return grid.compress(
            (data["<J*B>"] - data["<J*B> Redl"]),
        ) * jnp.sqrt(rho_weights / denominator)

    @property
    def helicity(self):
        """tuple: Type of quasi-symmetry (M, N)."""
        return self._helicity

    @helicity.setter
    def helicity(self, helicity):
        assert (
            (len(helicity) == 2)
            and (int(helicity[0]) == helicity[0])
            and (int(helicity[1]) == helicity[1])
        )
        assert helicity[0] == 1, "Redl bootstrap current model assumes helicity[0] == 1"
        if hasattr(self, "_helicity") and self._helicity != helicity:
            self._built = False
        self._helicity = helicity
