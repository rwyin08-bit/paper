#!/usr/bin/env python3
"""Entry point for PEBI black-oil simulations."""
import sys, os, argparse, yaml, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.black_oil_simulator import BlackOilSimulator, load_config

def generate_dfn(config, seed=0):
    np.random.seed(seed)
    d = config.get("dfn", {})
    n = d.get("n_fractures", 1247)
    kappa = d.get("fisher_concentration", 15.0)
    mu = np.radians(d.get("mean_strike_deg", 45.0))
    e = d.get("power_law_exponent", 2.3)
    Lmin, Lmax = d.get("length_min_m", 10), d.get("length_max_m", 500)
    fracs = []
    for _ in range(n):
        theta = np.random.vonmises(mu, kappa)
        L = (Lmin**(1-e) + np.random.random()*(Lmax**(1-e)-Lmin**(1-e)))**(1/(1-e))
        cx = np.random.uniform(0, config.get("domain_x_max", 4000))
        cy = np.random.uniform(0, config.get("domain_y_max", 3000))
        dx, dy = 0.5*L*np.cos(theta), 0.5*L*np.sin(theta)
        fracs.append(np.array([[cx-dx,cy-dy],[cx+dx,cy+dy]]))
    return fracs

def main():
    p = argparse.ArgumentParser(description="PEBI Black-Oil Shale Reservoir Simulator")
    p.add_argument("--config", "-c", default="config/base_case.yaml")
    p.add_argument("--output", "-o", default="results")
    p.add_argument("--dfn", action="store_true")
    p.add_argument("--ensemble", type=int, default=1)
    p.add_argument("--single-well", action="store_true")
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    config = load_config(args.config)
    if args.single_well: config.update(config.get("single_well", {}))
    os.makedirs(args.output, exist_ok=True)

    all_results = []
    for r in range(args.ensemble):
        if args.ensemble > 1: print(f"\nRealization {r+1}/{args.ensemble}")
        fracs = generate_dfn(config, seed=r) if args.dfn else None
        sim = BlackOilSimulator(config)
        t0 = time.time()
        mesh = sim.generate_mesh(fracs)
        if args.verbose:
            print(f"  Mesh: {time.time()-t0:.1f}s, K-orth: {mesh['k_orthogonality_fraction']*100:.1f}%, Cells: {mesh['n_cells']:,}")
        results = sim.run()
        results["mesh_stats"] = mesh
        results["realization"] = r
        all_results.append(results)

    if args.ensemble > 1:
        cum = [r.get("cumulative_oil",[0])[-1] for r in all_results]
        print(f"\nEnsemble (n={args.ensemble}): {np.mean(cum):.2f} +/- {np.std(cum):.2f}")

    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results: {args.output}/results.json")

if __name__ == "__main__":
    main()