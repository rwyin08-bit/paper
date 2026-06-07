#!/usr/bin/env python3
"""
pebi_mesh.py — Fracture-Conforming PEBI Grid Generator

Implements constrained Delaunay–Voronoi tessellation with hyperbolic-tangent
spacing function for black-oil shale-reservoir simulation.

Reference: Yin & Zhang (2024), Section 2.3
"""

import numpy as np
from scipy.spatial import Delaunay, Voronoi
from typing import List, Tuple, Optional


class PEBIMeshGenerator:
    """
    Generates fracture-conforming perpendicular-bisector (PEBI) grids
    via constrained Delaunay triangulation and Voronoi dual extraction.

    Parameters
    ----------
    h_n : float
        Near-fracture target spacing (m). Default: 1.0
    h_f : float
        Far-field target spacing (m). Default: 50.0
    alpha : float
        Tanh transition steepness (m⁻¹). Default: 0.5
    min_angle : float
        Minimum angle for Ruppert refinement (degrees). Default: 25.0
    lloyd_iterations : int
        Maximum Lloyd smoothing iterations. Default: 50
    lloyd_tolerance : float
        Convergence tolerance for Lloyd smoothing. Default: 1e-4 * h_n
    """

    def __init__(
        self,
        h_n: float = 1.0,
        h_f: float = 50.0,
        alpha: float = 0.5,
        min_angle: float = 25.0,
        lloyd_iterations: int = 50,
        lloyd_tolerance: float = None,
    ):
        self.h_n = h_n
        self.h_f = h_f
        self.alpha = alpha
        self.min_angle = min_angle
        self.lloyd_iterations = lloyd_iterations
        self.lloyd_tolerance = lloyd_tolerance or 1e-4 * h_n

        # Grid state
        self.generator_points = None
        self.triangulation = None
        self.voronoi = None
        self.fracture_traces = None
        self.k_orthogonality_fraction = None
        self.median_theta = None

    def spacing_function(self, x: np.ndarray) -> np.ndarray:
        """
        Hyperbolic-tangent spacing function (Eq. 9).

        h(x) = h_f + (h_n - h_f) * tanh(alpha * h_min(x))

        Parameters
        ----------
        x : np.ndarray of shape (n_points, 2)
            Point coordinates in the areal plane.

        Returns
        -------
        h : np.ndarray of shape (n_points,)
            Target element spacing at each point.
        """
        h_min = self._minimum_distance_to_fractures(x)
        return self.h_f + (self.h_n - self.h_f) * np.tanh(self.alpha * h_min)

    def _minimum_distance_to_fractures(self, x: np.ndarray) -> np.ndarray:
        """Compute minimum Euclidean distance from each point to fracture network."""
        if self.fracture_traces is None:
            return np.full(x.shape[0], self.h_f)

        dists = np.full(x.shape[0], np.inf)
        for trace in self.fracture_traces:
            # Distance from point to line segment (fracture trace)
            for i in range(len(trace) - 1):
                p1, p2 = trace[i], trace[i + 1]
                d = self._point_to_segment_distance(x, p1, p2)
                dists = np.minimum(dists, d)
        return dists

    @staticmethod
    def _point_to_segment_distance(
        points: np.ndarray, p1: np.ndarray, p2: np.ndarray
    ) -> np.ndarray:
        """Vectorized point-to-segment distance."""
        v = p2 - p1
        w = points - p1
        c1 = np.sum(w * v, axis=1)
        c2 = np.sum(v * v)
        t = np.clip(c1 / c2, 0, 1)
        projection = p1 + t[:, np.newaxis] * v
        return np.sqrt(np.sum((points - projection) ** 2, axis=1))

    def seed_generators(self, domain_bounds: Tuple[float, float, float, float]):
        """
        Stage 1: Constrained point seeding.

        Places generator points along fracture traces at spacing h_n
        and in the matrix region according to the spacing function.
        """
        x_min, x_max, y_min, y_max = domain_bounds

        # Uniform background grid at h_f spacing
        nx = int((x_max - x_min) / self.h_f) + 1
        ny = int((y_max - y_min) / self.h_f) + 1
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        background = np.column_stack([X.ravel(), Y.ravel()])

        # Refine near fractures
        h_vals = self.spacing_function(background)
        refined = background[h_vals < self.h_f * 0.9]

        # Add fracture-aligned points
        fracture_points = []
        if self.fracture_traces:
            for trace in self.fracture_traces:
                for i in range(len(trace) - 1):
                    seg_len = np.linalg.norm(trace[i + 1] - trace[i])
                    n_pts = max(2, int(seg_len / self.h_n))
                    t = np.linspace(0, 1, n_pts)[:, np.newaxis]
                    pts = trace[i] + t * (trace[i + 1] - trace[i])
                    fracture_points.append(pts)
            fracture_points = np.vstack(fracture_points)

        self.generator_points = np.vstack([refined, fracture_points]) if len(fracture_points) > 0 else refined
        return self.generator_points

    def build_triangulation(self):
        """
        Stage 2-3: Constrained Delaunay triangulation with Ruppert refinement.

        Constructs CDT with fracture traces as constrained edges.
        Inserts Steiner points at circumcenters of triangles violating
        the minimum angle bound or circumradius-to-local-spacing ratio.
        """
        if self.generator_points is None:
            raise ValueError("Call seed_generators() first.")

        self.triangulation = Delaunay(self.generator_points)

        # Ruppert refinement loop
        refined = True
        iteration = 0
        while refined and iteration < 100:
            refined = False
            iteration += 1

            for simplex in self.triangulation.simplices:
                tri_points = self.generator_points[simplex]
                angles = self._compute_angles(tri_points)

                if np.min(angles) < self.min_angle:
                    # Insert Steiner point at circumcenter
                    circumcenter = self._circumcenter(tri_points)
                    self.generator_points = np.vstack(
                        [self.generator_points, circumcenter]
                    )
                    refined = True
                    self.triangulation = Delaunay(self.generator_points)
                    break

        return self.triangulation

    @staticmethod
    def _compute_angles(triangle_points: np.ndarray) -> np.ndarray:
        """Compute interior angles of a triangle in degrees."""
        a, b, c = triangle_points
        ba, ca = a - b, a - c
        cb, ab = b - c, b - a
        ac, bc = c - a, c - b
        angles = np.array([
            np.arccos(np.dot(ba, ca) / (np.linalg.norm(ba) * np.linalg.norm(ca))),
            np.arccos(np.dot(cb, ab) / (np.linalg.norm(cb) * np.linalg.norm(ab))),
            np.arccos(np.dot(ac, bc) / (np.linalg.norm(ac) * np.linalg.norm(bc))),
        ])
        return np.degrees(angles)

    @staticmethod
    def _circumcenter(triangle_points: np.ndarray) -> np.ndarray:
        """Compute circumcenter of a triangle."""
        a, b, c = triangle_points
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-12:
            return np.mean(triangle_points, axis=0)
        ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) +
              (b[0]**2 + b[1]**2) * (c[1] - a[1]) +
              (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
        uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) +
              (b[0]**2 + b[1]**2) * (a[0] - c[0]) +
              (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
        return np.array([ux, uy])

    def extract_voronoi(self):
        """
        Stage 4: Voronoi dual extraction.

        Obtains the PEBI grid as the geometric dual of the refined CDT.
        """
        if self.triangulation is None:
            raise ValueError("Call build_triangulation() first.")
        self.voronoi = Voronoi(self.generator_points)
        return self.voronoi

    def lloyd_smoothing(self) -> int:
        """
        Stage 5: Lloyd smoothing.

        Iteratively relocates generator points to Voronoi cell centroids
        to improve cell aspect ratios and K-orthogonality.

        Returns
        -------
        n_iter : int
            Number of iterations performed.
        """
        if self.voronoi is None:
            raise ValueError("Call extract_voronoi() first.")

        for iteration in range(self.lloyd_iterations):
            max_displacement = 0.0
            new_points = self.generator_points.copy()

            for i, region_idx in enumerate(self.voronoi.point_region):
                region = self.voronoi.regions[region_idx]
                if not region or -1 in region:
                    continue
                vertices = self.voronoi.vertices[region]
                centroid = np.mean(vertices, axis=0)
                displacement = np.linalg.norm(centroid - self.generator_points[i])
                max_displacement = max(max_displacement, displacement)
                new_points[i] = centroid

            self.generator_points = new_points
            self.triangulation = Delaunay(self.generator_points)
            self.voronoi = Voronoi(self.generator_points)

            if max_displacement < self.lloyd_tolerance:
                return iteration + 1

        return self.lloyd_iterations

    def compute_k_orthogonality(self) -> Tuple[float, float]:
        """
        Compute K-orthogonality statistics (Eq. 11).

        Returns
        -------
        fraction : float
            Fraction of faces with theta <= 5 degrees.
        median_theta : float
            Median deviation angle in degrees.
        """
        if self.voronoi is None:
            raise ValueError("Call extract_voronoi() first.")

        angles = []
        for ridge_vertices, ridge_points in zip(
            self.voronoi.ridge_vertices, self.voronoi.ridge_points
        ):
            if -1 in ridge_vertices:
                continue  # skip boundary ridges

            i, j = ridge_points
            center_vec = self.generator_points[j] - self.generator_points[i]
            face_center = np.mean(self.voronoi.vertices[ridge_vertices], axis=0)
            face_normal = self._compute_face_normal(
                self.voronoi.vertices[ridge_vertices]
            )

            cos_theta = abs(np.dot(center_vec, face_normal)) / (
                np.linalg.norm(center_vec) * np.linalg.norm(face_normal) + 1e-15
            )
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_theta)))

        angles = np.array(angles)
        self.k_orthogonality_fraction = np.mean(angles <= 5.0)
        self.median_theta = np.median(angles)
        return self.k_orthogonality_fraction, self.median_theta

    @staticmethod
    def _compute_face_normal(vertices: np.ndarray) -> np.ndarray:
        """Compute outward normal of a Voronoi face."""
        if len(vertices) < 2:
            return np.array([0.0, 0.0])
        edge = vertices[1] - vertices[0]
        normal = np.array([-edge[1], edge[0]])
        return normal / (np.linalg.norm(normal) + 1e-15)

    def generate(
        self, domain_bounds: Tuple[float, float, float, float],
        fracture_traces: Optional[List[np.ndarray]] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Run the complete five-stage PEBI mesh generation pipeline.

        Returns
        -------
        stats : dict
            K-orthogonality fraction, median theta, cell count, aspect ratios.
        """
        self.fracture_traces = fracture_traces

        if verbose:
            print("Stage 1: Constrained point seeding...")
        self.seed_generators(domain_bounds)

        if verbose:
            print("Stage 2-3: Constrained Delaunay + Ruppert refinement...")
        self.build_triangulation()

        if verbose:
            print("Stage 4: Voronoi dual extraction...")
        self.extract_voronoi()

        if verbose:
            print("Stage 5: Lloyd smoothing...")
        n_lloyd = self.lloyd_smoothing()

        k_frac, med_theta = self.compute_k_orthogonality()

        stats = {
            "k_orthogonality_fraction": k_frac,
            "median_theta_deg": med_theta,
            "n_cells": len(self.generator_points),
            "n_lloyd_iterations": n_lloyd,
            "max_aspect_ratio": self._compute_max_aspect_ratio(),
        }

        if verbose:
            print(f"  K-orthogonality: {k_frac*100:.1f}% (theta <= 5 deg)")
            print(f"  Median theta: {med_theta:.1f} deg")
            print(f"  Cells: {stats['n_cells']:,}")

        return stats

    def _compute_max_aspect_ratio(self) -> float:
        """Compute maximum cell aspect ratio."""
        if self.voronoi is None:
            return 0.0
        max_ar = 0.0
        for region_idx in self.voronoi.point_region:
            region = self.voronoi.regions[region_idx]
            if not region or -1 in region:
                continue
            verts = self.voronoi.vertices[region]
            if len(verts) < 3:
                continue
            bbox = np.max(verts, axis=0) - np.min(verts, axis=0)
            ar = max(bbox) / (min(bbox) + 1e-15)
            max_ar = max(max_ar, ar)
        return max_ar