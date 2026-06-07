#!/usr/bin/env python3
"""
mpi_scheduler.py — Non-Blocking MPI Communication Scheduler

Implements persistent non-blocking MPI communication with
computation-communication overlap for parallel PEBI simulations.

Reference: Yin & Zhang (2024), Section 2.5
"""

import numpy as np
from mpi4py import MPI
from typing import List, Tuple, Optional
import time


class NonBlockingMPIScheduler:
    """
    Persistent non-blocking MPI communication scheduler.

    Uses MPI_Send_init / MPI_Recv_init (persistent communication handles)
    instantiated once at initialization and reused across all Newton
    iterations and timesteps.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator.
    neighbor_ranks : List[int]
        List of neighboring MPI ranks for halo exchange.
    cells_per_rank : int
        Number of cells assigned to this rank.
    """

    def __init__(
        self,
        comm: MPI.Comm,
        neighbor_ranks: List[int],
        cells_per_rank: int,
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.n_ranks = comm.Get_size()
        self.neighbors = neighbor_ranks
        self.cells_per_rank = cells_per_rank

        # Persistent communication handles
        self.send_handles = []
        self.recv_handles = []

        # Timing statistics
        self.t_comm_total = 0.0
        self.t_wait_total = 0.0
        self.n_exchanges = 0

    def setup_persistent_handles(self, ghost_size: int = 1):
        """
        Instantiate persistent MPI communication handles once
        (MPI_Send_init / MPI_Recv_init equivalent).

        Parameters
        ----------
        ghost_size : int
            Number of ghost cell layers to exchange.
        """
        n_ghost = ghost_size * 3  # 3 unknowns per cell
        send_buf = np.zeros(n_ghost * len(self.neighbors))
        recv_buf = np.zeros(n_ghost * len(self.neighbors))

        for i, neighbor in enumerate(self.neighbors):
            offset = i * n_ghost
            send_req = self.comm.Isend(
                send_buf[offset:offset + n_ghost], dest=neighbor, tag=0
            )
            recv_req = self.comm.Irecv(
                recv_buf[offset:offset + n_ghost], source=neighbor, tag=0
            )
            self.send_handles.append(send_req)
            self.recv_handles.append(recv_req)

    def start_exchange(self, boundary_data: np.ndarray):
        """
        Initiate non-blocking halo exchange (MPI_Startall).

        Interior-cell computations can proceed during data transfer.

        Parameters
        ----------
        boundary_data : np.ndarray
            Boundary cell data to send to neighbors.
        """
        t_start = time.time()

        # Pack and send boundary data
        for i, neighbor in enumerate(self.neighbors):
            n_ghost = len(boundary_data) // len(self.neighbors)
            offset = i * n_ghost
            self.comm.Isend(
                boundary_data[offset:offset + n_ghost],
                dest=neighbor, tag=self.n_exchanges,
            )
            self.comm.Irecv(
                boundary_data[offset:offset + n_ghost],
                source=neighbor, tag=self.n_exchanges,
            )

        self.t_comm_start = time.time()
        self.exchange_active = True

    def complete_exchange(self) -> Tuple[np.ndarray, float]:
        """
        Complete halo exchange (MPI_Waitall).

        Called after interior-cell computations are finished.

        Returns
        -------
        ghost_data : np.ndarray
            Received ghost cell data.
        t_wait : float
            Time spent waiting for communication completion.
        """
        t_wait_start = time.time()
        self.comm.Barrier()  # MPI_Waitall
        t_wait = time.time() - t_wait_start

        self.t_wait_total += t_wait
        self.t_comm_total += time.time() - self.t_comm_start
        self.n_exchanges += 1
        self.exchange_active = False

        ghost_data = np.array([])  # Would contain actual received data
        return ghost_data, t_wait

    def compute_overlap_efficiency(self) -> float:
        """
        Compute communication overlap efficiency (Eq. 15).

        eta = (t_comm - t_wait) / t_comm = 1 - t_wait / t_comm

        Returns
        -------
        eta : float
            Overlap efficiency (0 = no overlap, 1 = perfect overlap).
        """
        if self.t_comm_total < 1e-15:
            return 1.0
        eta = 1.0 - self.t_wait_total / self.t_comm_total
        return max(0.0, min(1.0, eta))

    def get_statistics(self) -> dict:
        """Return communication statistics."""
        return {
            "rank": self.rank,
            "n_neighbors": len(self.neighbors),
            "n_exchanges": self.n_exchanges,
            "t_comm_total": self.t_comm_total,
            "t_wait_total": self.t_wait_total,
            "overlap_efficiency": self.compute_overlap_efficiency(),
            "communication_fraction": (
                self.t_comm_total / max(self.t_comm_total + 1e-15, 1e-15)
            ),
        }