# Data Directory

This directory contains simulation input and output data.

## Quick Start
```bash
# Download SPE 10 dataset automatically
python download_spe10.py
```

## SPE 10 Dataset
The SPE 10 Comparative Solution Project dataset is publicly available from:
https://www.spe.org/web/csp/datasets/set02.htm

Files: `spe10_perm.dat` (permeability field), `spe10_phi.dat` (porosity field)

## Simulation Configuration
- Base case: five-well pad, 100 hydraulic-fracture stages
- Reservoir parameters: see Table 5 in manuscript
- DFN: 1,247 natural fractures (25 Monte Carlo realizations)
