# Fracture-Conforming PEBI Grids with Integrated CPR-SA-AMG Preconditioning and Non-Blocking MPI Communication for Scalable Black-Oil Simulation of Shale Reservoirs

Rongwang Yin¹, Shaowei Zhang²,*

¹ Basic Experiment and Training Center, Hefei University, Hefei 230601, P.R. China  
² School of Computer Engineering, Anhui Wenda University of Information Engineering, Hefei 231201, P.R. China

---

## Repository Structure

```
├── README.md                          # This file
├── manuscript/
│   ├── manuscript_revised.docx        # Revised manuscript (final version)
│   └── manuscript_original.docx       # Original submission
├── figures/
│   └── Fig1-12.*                      # Figure files (12 figures)
├── code/
│   └── ...                            # Reproducible simulation code
├── supplementary/
│   ├── highlights.docx                # Research highlights
│   ├── cover_letter.docx              # Cover letter
│   └── declaration_competing_interests.docx
└── data/
    └── ...                            # Simulation data / SPE10 inputs
```

## Abstract

The numerical simulation of hydraulically fractured shale-oil reservoirs confronts three interdependent computational bottlenecks: geometric mismatch at fracture-matrix interfaces, severe ill-conditioning of the Jacobian system arising from extreme permeability contrasts, and communication-bound parallel overhead at scale. This paper presents a black-oil simulation framework that treats mesh generation, pressure-system preconditioning, and MPI communication scheduling as a single co-design problem. The framework is built on fracture-conforming perpendicular-bisector (PEBI) grids constructed via constrained Delaunay-Voronoi tessellation within a layered 2.5D extrusion formulation. K-orthogonality is maintained across 97.1% of interior faces. The CPR-SA-AMG preconditioner reduces GMRES iterations from 312 to 21 (72.7% reduction in linear-solve time). On a 32-million-cell five-well pad problem with 100 hydraulic-fracture stages, the simulator achieves 82.0% strong-scaling efficiency on 512 MPI ranks (73% communication overlap) and 91.3% weak-scaling efficiency. Under 1,247 stochastically generated natural fractures, the spectral conditioning proxy increases by only 2.94+/-0.41x (Monte Carlo ensemble mean, n=25), compared with 18.6x and 11.4x for standard unstructured and Cartesian LGR grids; cumulative oil recovery increases by 21.8+/-4.3%. SPE 10 validation demonstrates a permeability-informed PEBI mesh of 0.52M cells tracks the 1.12M-cell Cartesian reference to within 0.93% cumulative oil deviation.

## Key Contributions

1. **Fracture-conforming PEBI mesh generation** with constrained Delaunay-Voronoi tessellation and hyperbolic-tangent spacing function (97.1% K-orthogonality)
2. **CPR-SA-AMG preconditioner** reducing GMRES iterations from 312 to 21
3. **Non-blocking MPI communication** with 73% communication overlap, 82.0% strong-scaling to 512 cores
4. **DFN robustness**: PEBI spectral conditioning degrades 6.5x slower than standard unstructured grids
5. **SPE 10 validation**: 0.93% accuracy at half the reference cell count

## Software

- PETSc 3.19 (https://petsc.org/)
- ParMETIS 4.0.3
- SLEPc 3.19
- Intel oneAPI 2023.1 / Intel MPI 2021.8

## License

MIT License
