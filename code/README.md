# PEBI Black-Oil Shale Reservoir Simulator

Open-source implementation of the fracture-conforming PEBI simulation framework
described in **Yin & Zhang (2024)**.

## Architecture

```
code/
├── src/
│   ├── __init__.py                  # Package entry
│   ├── pebi_mesh.py                 # PEBI mesh generator (Section 2.3)
│   ├── cpr_sa_amg.py                # CPR-SA-AMG preconditioner (Section 2.4)
│   ├── mpi_scheduler.py             # Non-blocking MPI scheduler (Section 2.5)
│   └── black_oil_simulator.py       # Main simulator
├── config/
│   ├── base_case.yaml               # Five-well pad (Table 5)
│   └── parameters.yaml              # Sensitivity parameters (Section 3.7)
├── docker/
│   ├── Dockerfile                   # Container definition
│   └── docker-compose.yml           # Multi-node deployment
├── scripts/
│   └── run_simulation.py            # Entry point
└── spe10/
    └── README.md                    # SPE 10 dataset instructions
```

## Quick Start

### Local Execution

```bash
# Install dependencies
pip install numpy scipy matplotlib pyyaml mpi4py petsc4py

# Run base case (five-well pad, 100 stages)
python scripts/run_simulation.py -c config/base_case.yaml

# Run with DFN (1,247 natural fractures)
python scripts/run_simulation.py -c config/base_case.yaml --dfn

# Run Monte Carlo ensemble (n=25 realizations)
python scripts/run_simulation.py -c config/base_case.yaml --dfn --ensemble 25

# Single-well validation
python scripts/run_simulation.py -c config/base_case.yaml --single-well
```

### Docker Execution

```bash
# Build image
docker build -t pebi-simulator -f docker/Dockerfile .

# Run base case
docker run --rm -v $(pwd)/results:/results \
    pebi-simulator -c config/base_case.yaml -o /results

# Multi-node (512 ranks, 8 nodes)
docker-compose -f docker/docker-compose.yml up pebi-cluster
```

### HPC Execution (MPI)

```bash
# Compile with Intel oneAPI + PETSc
module load intel/2023.1 mpi/2021.8 petsc/3.19

# Run on 512 ranks
mpirun -np 512 python scripts/run_simulation.py \
    -c config/base_case.yaml --parallel
```

## Key Results

| Metric | Value | Reference |
|--------|-------|-----------|
| K-orthogonality | 97.1% (theta <= 5 deg) | Section 2.3, Fig. 4 |
| GMRES iterations | 21 (vs. 312 for ILU) | Section 3.4, Table 9 |
| Strong scaling (512 cores) | 82.0% | Section 3.5, Fig. 9 |
| Weak scaling (512 cores) | 91.3% | Section 3.5, Table 11 |
| Communication overlap | 73% (eta = 0.73) | Section 2.5, Eq. 15 |
| DFN condition number | 2.94 +/- 0.41x (n=25) | Section 3.6, Table 12b |
| SPE 10 accuracy | 0.93% deviation | Section 3.2, Table 7 |

## Dependencies

- Python 3.8+
- NumPy, SciPy, Matplotlib, PyYAML
- MPI4Py (for parallel execution)
- PETSc 3.19+ (with ParMETIS)
- SLEPc 3.19+

## Reproducibility

All numerical experiments in Sections 3.1-3.8 can be reproduced using:

```bash
# Section 3.1: Single-phase validation
python scripts/run_simulation.py -c config/base_case.yaml --single-phase

# Section 3.4: Solver comparison
python scripts/run_simulation.py -c config/base_case.yaml --solver-comparison

# Section 3.5: Parallel scaling
mpirun -np 64,128,256,512,1024 python scripts/run_simulation.py -c config/base_case.yaml --scaling

# Section 3.6: DFN robustness
python scripts/run_simulation.py -c config/base_case.yaml --dfn --ensemble 25

# Section 3.7: Sensitivity analysis
python scripts/run_simulation.py -c config/base_case.yaml --sensitivity
```

## License

MIT License

## Citation

```
Yin, R., Zhang, S. Fracture-Conforming PEBI Grids with Integrated
CPR-SA-AMG Preconditioning and Non-Blocking MPI Communication for
Scalable Black-Oil Simulation of Shale Reservoirs. (2024)
```