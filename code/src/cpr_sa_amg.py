#!/usr/bin/env python3
"""
cpr_sa_amg.py — CPR–SA-AMG Preconditioner

Implements the Constrained Pressure Residual preconditioner accelerated
by Smoothed-Aggregation Algebraic Multigrid for black-oil systems.

Reference: Yin & Zhang (2024), Section 2.4
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from typing import Tuple, Optional


class CPRSAAMGPreconditioner:
    """
    Two-stage preconditioner: SA-AMG V-cycle on pressure block (Stage 1)
    followed by Block-ILU(0) global smoothing (Stage 2).

    Parameters
    ----------
    n_cells : int
        Number of computational cells.
    strong_threshold : float
        SA-AMG strong-coupling threshold (epsilon_str). Default: 0.25
    damping_omega : float
        Damped Jacobi smoothing parameter. Default: 0.7
    max_coarse_size : int
        Maximum size of coarsest grid. Default: 500
    gmres_restart : int
        GMRES restart dimension. Default: 30
    gmres_tolerance : float
        GMRES relative residual tolerance. Default: 1e-6
    """

    def __init__(
        self,
        n_cells: int,
        strong_threshold: float = 0.25,
        damping_omega: float = 0.7,
        max_coarse_size: int = 500,
        gmres_restart: int = 30,
        gmres_tolerance: float = 1e-6,
    ):
        self.n_cells = n_cells
        self.n_unknowns = 3 * n_cells  # p, S_o, p_b per cell
        self.epsilon_str = strong_threshold
        self.omega = damping_omega
        self.max_coarse_size = max_coarse_size
        self.gmres_restart = gmres_restart
        self.gmres_tolerance = gmres_tolerance

        # AMG hierarchy
        self.levels = []
        self.prolongation = None
        self.restriction = None

    def extract_pressure_block(self, A: sparse.csr_matrix) -> sparse.csr_matrix:
        """
        Extract the pressure-pressure block A_pp from the full Jacobian.

        The full system is ordered as [p, S_o, p_b] per cell.
        A_pp contains entries A[i,j] where i%3==0 and j%3==0.
        """
        n = self.n_cells
        p_indices = np.arange(0, self.n_unknowns, 3)
        A_pp = A[p_indices][:, p_indices]
        return A_pp

    def build_amg_hierarchy(self, A_pp: sparse.csr_matrix):
        """
        Build SA-AMG hierarchy using smoothed aggregation.

        Constructs coarse-grid operators through:
        1. Strength-of-connection based on epsilon_str
        2. Tentative prolongator from aggregates
        3. Damped Jacobi smoothing: P = (I - omega*D^{-1}*A_pp) * P_0 (Eq. 14)
        """
        self.levels = [A_pp]
        n = A_pp.shape[0]

        while n > self.max_coarse_size:
            A_fine = self.levels[-1]

            # Strength matrix
            D_diag = A_fine.diagonal()
            D_inv = sparse.diags(1.0 / (np.abs(D_diag) + 1e-15))
            S = np.abs(D_inv @ A_fine) >= self.epsilon_str
            S.setdiag(1.0)

            # Aggregate nodes using greedy algorithm
            aggregates = self._greedy_aggregation(S)

            # Build tentative prolongator P_0
            n_agg = len(np.unique(aggregates))
            P_0 = sparse.lil_matrix((n, n_agg))
            for i, agg_id in enumerate(aggregates):
                P_0[i, agg_id] = 1.0
            P_0 = P_0.tocsr()

            # Damped Jacobi smoothing (Eq. 14)
            D = sparse.diags(D_diag)
            I = sparse.eye(n, format="csr")
            P = (I - self.omega * sparse.linalg.inv(D) @ A_fine) @ P_0

            # Galerkin coarse-grid operator
            R = P.T
            A_coarse = R @ A_fine @ P

            self.levels.append(A_coarse)
            n = A_coarse.shape[0]

        self.prolongation = P
        self.restriction = P.T

    @staticmethod
    def _greedy_aggregation(S: sparse.csr_matrix) -> np.ndarray:
        """Greedy algorithm for aggregating strongly connected nodes."""
        n = S.shape[0]
        aggregates = -np.ones(n, dtype=int)
        agg_id = 0
        unassigned = np.ones(n, dtype=bool)

        for i in range(n):
            if not unassigned[i]:
                continue
            # Form aggregate: node i plus its strong neighbors
            neighbors = S[i].indices
            for j in neighbors:
                if unassigned[j]:
                    aggregates[j] = agg_id
                    unassigned[j] = False
            aggregates[i] = agg_id
            unassigned[i] = False
            agg_id += 1

        return aggregates

    def v_cycle(self, b: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform one SA-AMG V-cycle.

        1. Pre-smoothing (damped Jacobi)
        2. Restrict residual to coarse grid
        3. Recursive solve / direct solve at coarsest level
        4. Prolongate correction
        5. Post-smoothing
        """
        if not self.levels:
            raise ValueError("Call build_amg_hierarchy() first.")

        x = np.zeros_like(b) if x0 is None else x0.copy()

        # Pre-smoothing (2 Jacobi iterations)
        A = self.levels[0]
        D_inv = 1.0 / (A.diagonal() + 1e-15)
        for _ in range(2):
            r = b - A @ x
            x += self.omega * D_inv * r

        # Restrict residual
        r = b - A @ x
        r_coarse = self.restriction @ r

        # Coarse-grid solve
        if len(self.levels) == 1:
            x_coarse = spla.spsolve(self.levels[-1], r_coarse)
        else:
            A_coarse = self.levels[-1]
            x_coarse = spla.spsolve(A_coarse, r_coarse)

        # Prolongate correction
        x += self.prolongation @ x_coarse

        # Post-smoothing (2 Jacobi iterations)
        for _ in range(2):
            r = b - A @ x
            x += self.omega * D_inv * r

        return x

    def apply_cpr(
        self, A: sparse.csr_matrix, b: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Apply the full two-stage CPR preconditioner within GMRES.

        Stage 1: SA-AMG V-cycle on pressure block A_pp.
        Stage 2: Block-ILU(0) global smoothing pass.

        Returns
        -------
        x : np.ndarray
            Solution vector.
        n_iterations : int
            Number of GMRES iterations.
        residual : float
            Final relative residual norm.
        """
        # Extract pressure block
        A_pp = self.extract_pressure_block(A)

        # Build AMG hierarchy
        self.build_amg_hierarchy(A_pp)

        # Stage 1: Pressure correction via SA-AMG
        n = self.n_cells
        b_p = b[0::3]  # Pressure components of RHS
        delta_p = self.v_cycle(b_p)

        # Stage 2: Block-ILU(0) on full system
        M_ilu = spla.spilu(A, drop_tol=0.0, fill_factor=0.0)

        # Combine stages in GMRES
        def preconditioner(v):
            # First apply ILU
            w = M_ilu.solve(v)
            # Then pressure correction
            w_p = self.v_cycle(v[0::3])
            w[0::3] += w_p - v[0::3] * 0
            return w

        M = spla.LinearOperator(
            (self.n_unknowns, self.n_unknowns),
            matvec=preconditioner,
        )

        # Solve with GMRES
        x0 = np.zeros(self.n_unknowns) if x0 is None else x0
        x, info = spla.gmres(
            A, b, x0=x0,
            M=M,
            restart=self.gmres_restart,
            tol=self.gmres_tolerance,
            maxiter=500,
        )

        residual = np.linalg.norm(b - A @ x) / (np.linalg.norm(b) + 1e-15)
        n_iterations = info if info > 0 else 500

        return x, n_iterations, residual

    def compute_diagonal_dominance(self, A: sparse.csr_matrix) -> float:
        """
        Compute mean diagonal dominance ratio.

        delta_i = |a_ii| / sum_{j!=i} |a_ij|
        """
        A_abs = np.abs(A)
        diag = A_abs.diagonal()
        row_sum = np.array(A_abs.sum(axis=1)).flatten()
        off_diag_sum = row_sum - diag
        ratios = diag / (off_diag_sum + 1e-15)
        return float(np.mean(ratios))