#!/usr/bin/env python3
"""
black_oil_simulator.py — Main Black-Oil Shale Reservoir Simulator

Integrates fracture-conforming PEBI grids, CPR-SA-AMG preconditioning,
and non-blocking MPI communication for scalable black-oil simulation.

Reference: Yin & Zhang (2024)
"""

import numpy as np
from scipy import sparse
from typing import Dict, Optional, Tuple
import time
import yaml

from pebi_mesh import PEBIMeshGenerator
from cpr_sa_amg import CPRSAAMGPreconditioner
from mpi_scheduler import NonBlockingMPIScheduler

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False


class BlackOilSimulator:
    """
    Fully implicit black-oil simulator for fractured shale reservoirs.

    Parameters
    ----------
    config : dict
        Simulation configuration including reservoir properties,
        well controls, and solver parameters.
    """

    def __init__(self, config: dict):
        self.config = config

        # Reservoir parameters
        self.nx = config.get("nx", 100)
        self.ny = config.get("ny", 100)
        self.nz = config.get("nz", 1)
        self.n_cells = self.nx * self.ny * self.nz

        # Physical properties
        self.phi = config.get("porosity", 0.07)
        self.k_matrix = config.get("matrix_permeability_md", 0.008)
        self.c_r = config.get("rock_compressibility_psi", 5e-6)
        self.p_init = config.get("initial_pressure_mpa", 35.0)
        self.p_bubble = config.get("bubble_point_pressure_mpa", 25.0)
        self.S_oi = config.get("initial_oil_saturation", 0.65)
        self.S_wi = config.get("initial_water_saturation", 0.35)

        # Fluid properties
        self.mu_o = config.get("oil_viscosity_cp", 0.35)
        self.B_o = config.get("oil_fvf", 1.35)
        self.R_s = config.get("solution_gor", 120.0)

        # Fracture properties
        self.fracture_half_length = config.get("fracture_half_length_m", 150.0)
        self.fracture_conductivity = config.get("fracture_conductivity_md_m", 500.0)
        self.fracture_aperture = config.get("fracture_aperture_mm", 3.0)
        self.n_stages = config.get("n_hydraulic_stages", 100)
        self.n_wells = config.get("n_wells", 5)

        # Simulation control
        self.sim_time_days = config.get("simulation_time_years", 10.0) * 365.0
        self.dt_max_days = config.get("max_timestep_days", 1.0)
        self.newton_tolerance = config.get("newton_tolerance", 1e-4)

        # Solver
        solver_cfg = config.get("solver", {})
        self.preconditioner = CPRSAAMGPreconditioner(
            n_cells=self.n_cells,
            strong_threshold=solver_cfg.get("sa_amg_epsilon_str", 0.25),
            damping_omega=solver_cfg.get("sa_amg_omega", 0.7),
            max_coarse_size=solver_cfg.get("sa_amg_max_coarse", 500),
            gmres_restart=solver_cfg.get("gmres_restart", 30),
            gmres_tolerance=solver_cfg.get("gmres_tolerance", 1e-6),
        )

        # MPI
        if HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.n_ranks = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.n_ranks = 1

        # State variables (per cell: [p, S_o, p_b])
        self.state = np.zeros(3 * self.n_cells)
        self.state[0::3] = self.p_init  # pressure
        self.state[1::3] = self.S_oi    # oil saturation
        self.state[2::3] = self.p_bubble  # bubble-point pressure

        # Mesh
        self.mesh = None
        self.mesh_stats = None

    def generate_mesh(self, fracture_network: Optional[list] = None):
        """Generate fracture-conforming PEBI mesh."""
        mesh_gen = PEBIMeshGenerator(
            h_n=self.config.get("mesh", {}).get("h_n", 1.0),
            h_f=self.config.get("mesh", {}).get("h_f", 50.0),
            alpha=self.config.get("mesh", {}).get("alpha", 0.5),
        )

        domain = (
            self.config.get("domain_x_min", 0),
            self.config.get("domain_x_max", 4000),
            self.config.get("domain_y_min", 0),
            self.config.get("domain_y_max", 3000),
        )

        self.mesh_stats = mesh_gen.generate(
            domain, fracture_traces=fracture_network, verbose=(self.rank == 0)
        )
        self.mesh = mesh_gen
        return self.mesh_stats

    def assemble_jacobian(self) -> sparse.csr_matrix:
        """
        Assemble the fully implicit Jacobian matrix.

        Returns block-structured system A_{3Nc x 3Nc} with
        approximately 21 nonzero entries per row (7-point stencil
        x 3 unknowns per cell).
        """
        n = self.n_cells
        N = 3 * n
        A = sparse.lil_matrix((N, N))

        for c in range(n):
            i3 = 3 * c  # base index for cell c

            # Pressure equation row
            A[i3, i3] = self.phi * self.c_r / self.B_o  # diagonal
            A[i3, i3 + 1] = -self.phi / (self.B_o**2)   # d/dS_o
            A[i3, i3 + 2] = 0.0                           # d/dp_b

            # Saturation equation row
            A[i3 + 1, i3] = 0.0
            A[i3 + 1, i3 + 1] = 1.0
            A[i3 + 1, i3 + 2] = 0.0

            # Bubble-point / GOR equation row
            A[i3 + 2, i3] = 0.0
            A[i3 + 2, i3 + 1] = 0.0
            A[i3 + 2, i3 + 2] = 1.0

            # Off-diagonal: transmissibility connections
            for neighbor in self._get_neighbors(c):
                j3 = 3 * neighbor
                T = self._compute_transmissibility(c, neighbor)
                A[i3, j3] = -T
                A[j3, i3] = -T
                A[i3, i3] += T
                A[j3, j3] += T

        return A.tocsr()

    def _get_neighbors(self, cell_idx: int) -> list:
        """Get neighboring cell indices for a structured grid."""
        i = cell_idx % self.nx
        j = (cell_idx // self.nx) % self.ny
        neighbors = []
        if i > 0: neighbors.append(cell_idx - 1)
        if i < self.nx - 1: neighbors.append(cell_idx + 1)
        if j > 0: neighbors.append(cell_idx - self.nx)
        if j < self.ny - 1: neighbors.append(cell_idx + self.nx)
        return neighbors

    def _compute_transmissibility(self, cell_i: int, cell_j: int) -> float:
        """Compute inter-cell transmissibility (Eq. 7)."""
        # Harmonic average of half-transmissibilities
        k = self.k_matrix * 9.869233e-16  # mD -> m^2
        dx = (self.config.get("domain_x_max", 4000) -
              self.config.get("domain_x_min", 0)) / self.nx
        A_face = dx * 1.0  # unit thickness
        T_half = k * A_face / (dx * self.mu_o * self.B_o)
        return 0.5 * T_half  # harmonic mean for uniform grid

    def compute_residual(self) -> np.ndarray:
        """Compute nonlinear residual vector R."""
        R = np.zeros(3 * self.n_cells)
        # Accumulation and flux terms
        for c in range(self.n_cells):
            i3 = 3 * c
            # Mass balance residual (simplified)
            R[i3] = 0.0      # pressure residual
            R[i3 + 1] = 0.0  # saturation residual
            R[i3 + 2] = 0.0  # GOR residual
        return R

    def newton_step(self) -> Tuple[np.ndarray, int, float]:
        """Perform one Newton-Raphson iteration."""
        A = self.assemble_jacobian()
        R = self.compute_residual()

        # Solve A * dx = -R using CPR-SA-AMG preconditioned GMRES
        dx, n_iter, residual = self.preconditioner.apply_cpr(
            A, -R, x0=None
        )

        # Update state
        self.state += dx

        # Apply physical bounds
        self.state[0::3] = np.clip(self.state[0::3], 1.0, 100.0)  # pressure
        self.state[1::3] = np.clip(self.state[1::3], 0.0, 1.0)     # saturation

        return dx, n_iter, residual

    def run(self) -> dict:
        """
        Run the full simulation.

        Returns
        -------
        results : dict
            Simulation results including production data and solver statistics.
        """
        if self.rank == 0:
            print(f"Black-Oil Simulation: {self.n_cells:,} cells, "
                  f"{self.n_ranks} MPI ranks")
            print(f"Simulation period: {self.sim_time_days:.0f} days")
            print(f"Max timestep: {self.dt_max_days} days")

        results = {
            "time_days": [],
            "cumulative_oil": [],
            "gmres_iterations": [],
            "newton_iterations": [],
            "wall_clock_time": [],
        }

        t_sim_start = time.time()
        t_current = 0.0
        timestep = 0

        while t_current < self.sim_time_days:
            dt = min(self.dt_max_days, self.sim_time_days - t_current)
            t_current += dt
            timestep += 1

            # Newton-Raphson loop
            for newton_iter in range(20):
                dx, n_gmres, resid = self.newton_step()

                if np.linalg.norm(dx) < self.newton_tolerance:
                    break

            # Record results
            results["time_days"].append(t_current)
            results["cumulative_oil"].append(0.0)  # placeholder
            results["gmres_iterations"].append(n_gmres)
            results["newton_iterations"].append(newton_iter + 1)

            if self.rank == 0 and timestep % 100 == 0:
                print(f"  Day {t_current:.0f}/{self.sim_time_days:.0f}")

        t_total = time.time() - t_sim_start
        results["total_wall_time"] = t_total
        results["n_timesteps"] = timestep

        if self.rank == 0:
            print(f"Simulation complete: {t_total:.1f} s wall time")

        return results


def load_config(config_path: str) -> dict:
    """Load simulation configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Black-Oil Shale Reservoir Simulator"
    )
    parser.add_argument(
        "--config", "-c", default="config/base_case.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", "-o", default="results",
        help="Output directory for results"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    sim = BlackOilSimulator(config)
    sim.generate_mesh()
    results = sim.run()

    # Save results
    import json
    with open(f"{args.output}/results.json", "w") as f:
        json.dump({k: v if isinstance(v, list) else v
                   for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()