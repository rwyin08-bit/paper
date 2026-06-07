"""
Fracture-Conforming PEBI Simulation Framework for Black-Oil Shale Reservoirs.

Reference: Yin & Zhang (2024)
"""

from .pebi_mesh import PEBIMeshGenerator
from .cpr_sa_amg import CPRSAAMGPreconditioner
from .mpi_scheduler import NonBlockingMPIScheduler
from .black_oil_simulator import BlackOilSimulator

__version__ = "1.0.0"
__all__ = [
    "PEBIMeshGenerator",
    "CPRSAAMGPreconditioner",
    "NonBlockingMPIScheduler",
    "BlackOilSimulator",
]