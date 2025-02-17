"""Make sure that the grids are behaving the way I expect them to."""

import numpy as np
from desc.grid import QuadratureGrid

def test_quadrature_grid():
    NFP = 5
    grid = QuadratureGrid(
        L=4,
        M=4,
        N=3,
        NFP=NFP,
    )

    rho_grid = np.array(sorted(list(set(grid.nodes[:, 0]))))
    print("Rho grid points:", rho_grid)
    rho_spacing = grid.compress(grid.spacing[:, 0])
    print("grid.compress(grid.spacing[:, 0]):", rho_spacing)
    print("sum(rho_spacing):", sum(rho_spacing))
    print("sum(rho * rho_spacing):", sum(rho_grid * rho_spacing))
    print("nodes:")
    print(grid.nodes)

    print("weights:", grid.weights)
    print("# of nodes:", len(grid.weights))
    print(dir(grid))
    vars = ["coordinates", "num_poloidal", "num_rho", "num_theta", "num_zeta", "unique_rho_idx", "inverse_rho_idx"]
    for var in vars:
        print(f"{var}: {getattr(grid, var)}")

    rho_weights = grid.weights[grid.unique_rho_idx]
    print("rho_weights:", rho_weights)
    print("sum(grid.weights) / (4 pi^2):", sum(grid.weights) / (4 * np.pi**2))
    print("sum(rho * weights) / (4 pi^2):", sum(grid.nodes[:, 0] * grid.weights) / (4 * np.pi**2))
    expected_rho_spacing = rho_weights / (4 * np.pi**2) * grid.num_theta * grid.num_zeta
    print("Expected rho_spacing:", expected_rho_spacing)
    print("rho_spacing:", rho_spacing)

    np.testing.assert_allclose(rho_spacing, expected_rho_spacing)
    np.testing.assert_allclose(sum(grid.nodes[:, 0] * grid.weights) / (4 * np.pi**2), 0.5)
    np.testing.assert_allclose(sum(rho_grid * rho_spacing), 0.5)
