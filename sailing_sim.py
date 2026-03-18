"""
Sailing Simulator — Optimal Upwind Path Visualization

Computes the fastest tacking path for a dinghy sailing upwind from point X
to point Y, then animates the boat following that path in a browser.

Supports Laser and 420 boat types with realistic polar speed data.

Usage:
    python3 sailing_sim.py          # opens browser at http://localhost:8420
    python3 sailing_sim.py --port 9000
"""

from __future__ import annotations

import hashlib
import http.server
import json
import math
import socketserver
import sys
import webbrowser
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


# ============================================================
# Section 1: Constants
# ============================================================

KNOTS_TO_NM_PER_SEC = 1.0 / 3600.0


# ============================================================
# Section 2: Dataclasses
# ============================================================

@dataclass(frozen=True)
class SimConfig:
    """Top-level simulation configuration."""
    boat_type: str = "laser"
    wind_speed_kts: float = 12.0
    wind_direction_deg: float = 0.0  # wind FROM north (blows southward)
    start_x: float = 0.0
    start_y: float = 0.0
    mark_x: float = 0.0
    mark_y: float = 1.0
    finish_x: float = 0.0
    finish_y: float = 0.0
    dt_seconds: float = 1.0


@dataclass(frozen=True)
class WindState:
    """Wind conditions at a point in time/space."""
    speed_kts: float
    direction_deg: float


@dataclass
class Waypoint:
    """Single point on the pre-computed path."""
    x: float
    y: float
    heading_deg: float
    tack: str
    speed_kts: float
    vmg_kts: float
    elapsed_seconds: float
    leg: str = "upwind"


@dataclass
class SailingPath:
    """Pre-computed optimal path from start to target."""
    waypoints: list[Waypoint]
    total_distance_nm: float
    total_time_seconds: float
    optimal_twa_deg: float
    optimal_vmg_kts: float
    optimal_speed_kts: float
    n_tacks: int
    legs: list[dict]


# ============================================================
# Section 3: Polar Speed Model
# ============================================================

LASER_POLAR_DATA: dict[int, list[tuple[float, float]]] = {
    6: [
        (0, 0.0), (20, 0.0), (25, 0.5), (30, 1.2), (35, 2.0),
        (40, 2.6), (45, 2.9), (52, 3.1), (60, 3.3), (70, 3.5),
        (80, 3.6), (90, 3.6), (100, 3.5), (110, 3.4), (120, 3.2),
        (130, 2.9), (140, 2.6), (150, 2.3), (160, 1.9), (170, 1.6), (180, 1.4),
    ],
    8: [
        (0, 0.0), (20, 0.0), (25, 0.8), (30, 1.5), (35, 2.8),
        (40, 3.5), (45, 3.8), (52, 4.0), (60, 4.3), (70, 4.5),
        (80, 4.6), (90, 4.7), (100, 4.7), (110, 4.6), (120, 4.4),
        (130, 4.1), (140, 3.7), (150, 3.3), (160, 2.8), (170, 2.4), (180, 2.0),
    ],
    12: [
        (0, 0.0), (20, 0.0), (25, 1.2), (30, 2.0), (35, 3.5),
        (40, 4.3), (45, 4.7), (52, 5.0), (60, 5.2), (70, 5.4),
        (80, 5.5), (90, 5.5), (100, 5.4), (110, 5.3), (120, 5.0),
        (130, 4.6), (140, 4.2), (150, 3.7), (160, 3.2), (170, 2.8), (180, 2.4),
    ],
    15: [
        (0, 0.0), (20, 0.0), (25, 1.5), (30, 2.5), (35, 3.8),
        (40, 4.6), (45, 5.0), (52, 5.3), (60, 5.5), (70, 5.7),
        (80, 5.8), (90, 5.8), (100, 5.7), (110, 5.5), (120, 5.2),
        (130, 4.8), (140, 4.3), (150, 3.8), (160, 3.3), (170, 2.9), (180, 2.5),
    ],
    20: [
        (0, 0.0), (20, 0.0), (25, 1.8), (30, 2.8), (35, 4.0),
        (40, 4.8), (45, 5.2), (52, 5.5), (60, 5.7), (70, 5.9),
        (80, 6.0), (90, 6.0), (100, 5.9), (110, 5.7), (120, 5.4),
        (130, 5.0), (140, 4.5), (150, 4.0), (160, 3.5), (170, 3.0), (180, 2.6),
    ],
}

# 420: flatter upwind polar (less speed gain from bearing off), faster on reaches (trapeze/spinnaker)
# Key difference from Laser: optimal VMG angle is tighter (~39-41° vs ~43-45°)
FOUR_TWENTY_POLAR_DATA: dict[int, list[tuple[float, float]]] = {
    6: [
        (0, 0.0), (20, 0.0), (25, 0.4), (30, 1.2), (35, 2.2),
        (40, 2.7), (45, 2.8), (52, 2.9), (60, 3.2), (70, 3.5),
        (80, 3.7), (90, 3.8), (100, 3.8), (110, 3.7), (120, 3.5),
        (130, 3.2), (140, 2.9), (150, 2.5), (160, 2.1), (170, 1.7), (180, 1.5),
    ],
    8: [
        (0, 0.0), (20, 0.0), (25, 0.8), (30, 1.6), (35, 3.0),
        (40, 3.5), (45, 3.6), (52, 3.8), (60, 4.2), (70, 4.6),
        (80, 4.8), (90, 5.0), (100, 5.0), (110, 4.9), (120, 4.7),
        (130, 4.4), (140, 4.0), (150, 3.5), (160, 3.0), (170, 2.5), (180, 2.1),
    ],
    12: [
        (0, 0.0), (20, 0.0), (25, 1.2), (30, 2.2), (35, 3.7),
        (40, 4.3), (45, 4.4), (52, 4.6), (60, 5.1), (70, 5.5),
        (80, 5.7), (90, 5.8), (100, 5.8), (110, 5.7), (120, 5.4),
        (130, 5.0), (140, 4.5), (150, 4.0), (160, 3.4), (170, 2.9), (180, 2.5),
    ],
    15: [
        (0, 0.0), (20, 0.0), (25, 1.5), (30, 2.6), (35, 4.0),
        (40, 4.6), (45, 4.7), (52, 4.9), (60, 5.4), (70, 5.7),
        (80, 5.9), (90, 6.0), (100, 6.0), (110, 5.9), (120, 5.6),
        (130, 5.2), (140, 4.7), (150, 4.2), (160, 3.6), (170, 3.1), (180, 2.7),
    ],
    20: [
        (0, 0.0), (20, 0.0), (25, 1.8), (30, 2.9), (35, 4.2),
        (40, 4.8), (45, 4.9), (52, 5.1), (60, 5.6), (70, 5.9),
        (80, 6.1), (90, 6.2), (100, 6.2), (110, 6.1), (120, 5.8),
        (130, 5.4), (140, 4.9), (150, 4.4), (160, 3.8), (170, 3.3), (180, 2.9),
    ],
}

# J/24: keelboat with steeper upwind polar (more speed gain from bearing off),
# so optimal VMG angle is wider (~46-48°). Higher top speeds, heavier displacement.
J24_POLAR_DATA: dict[int, list[tuple[float, float]]] = {
    6: [
        (0, 0.0), (20, 0.0), (25, 0.3), (30, 1.0), (35, 1.8),
        (40, 2.4), (45, 3.0), (52, 3.5), (60, 3.8), (70, 4.1),
        (80, 4.3), (90, 4.4), (100, 4.3), (110, 4.1), (120, 3.8),
        (130, 3.4), (140, 3.0), (150, 2.6), (160, 2.2), (170, 1.8), (180, 1.5),
    ],
    8: [
        (0, 0.0), (20, 0.0), (25, 0.5), (30, 1.3), (35, 2.4),
        (40, 3.2), (45, 3.9), (52, 4.5), (60, 4.9), (70, 5.3),
        (80, 5.5), (90, 5.6), (100, 5.5), (110, 5.3), (120, 5.0),
        (130, 4.6), (140, 4.1), (150, 3.6), (160, 3.0), (170, 2.5), (180, 2.1),
    ],
    12: [
        (0, 0.0), (20, 0.0), (25, 0.8), (30, 1.7), (35, 3.0),
        (40, 4.0), (45, 4.8), (52, 5.5), (60, 5.9), (70, 6.2),
        (80, 6.4), (90, 6.5), (100, 6.4), (110, 6.2), (120, 5.8),
        (130, 5.3), (140, 4.7), (150, 4.1), (160, 3.5), (170, 2.9), (180, 2.5),
    ],
    15: [
        (0, 0.0), (20, 0.0), (25, 1.0), (30, 2.0), (35, 3.3),
        (40, 4.3), (45, 5.2), (52, 5.9), (60, 6.3), (70, 6.6),
        (80, 6.8), (90, 6.9), (100, 6.8), (110, 6.5), (120, 6.1),
        (130, 5.6), (140, 5.0), (150, 4.4), (160, 3.7), (170, 3.1), (180, 2.7),
    ],
    20: [
        (0, 0.0), (20, 0.0), (25, 1.2), (30, 2.3), (35, 3.5),
        (40, 4.5), (45, 5.4), (52, 6.1), (60, 6.5), (70, 6.8),
        (80, 7.0), (90, 7.1), (100, 7.0), (110, 6.7), (120, 6.3),
        (130, 5.8), (140, 5.2), (150, 4.6), (160, 3.9), (170, 3.3), (180, 2.8),
    ],
}

BOAT_POLARS: dict[str, dict] = {
    "laser": LASER_POLAR_DATA,
    "420": FOUR_TWENTY_POLAR_DATA,
    "j24": J24_POLAR_DATA,
}

# Cache built interpolators per boat type
_interp_cache: dict[str, RegularGridInterpolator] = {}


def build_polar_interpolator(polar_data: dict) -> RegularGridInterpolator:
    """Build 2D interpolator: (TWS, TWA) -> boat speed in knots."""
    tws_values = sorted(polar_data.keys())
    twa_values = [entry[0] for entry in polar_data[tws_values[0]]]
    speeds = np.array(
        [[spd for _, spd in polar_data[tws]] for tws in tws_values]
    )
    return RegularGridInterpolator(
        (np.array(tws_values, dtype=float), np.array(twa_values, dtype=float)),
        speeds,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )


def get_interpolator(boat_type: str) -> RegularGridInterpolator:
    """Get or build the polar interpolator for a boat type."""
    if boat_type not in _interp_cache:
        _interp_cache[boat_type] = build_polar_interpolator(BOAT_POLARS[boat_type])
    return _interp_cache[boat_type]


def get_boat_speed(tws: float, twa_deg: float, interp: RegularGridInterpolator) -> float:
    """Look up interpolated boat speed for given TWS and TWA (0-180)."""
    twa_abs = abs(twa_deg) % 360
    if twa_abs > 180:
        twa_abs = 360 - twa_abs
    tws_clamped = np.clip(tws, 6.0, 20.0)
    return float(interp((tws_clamped, twa_abs)))


# ============================================================
# Section 4: Spatial Wind Model
# ============================================================

def make_wind_seed(config: SimConfig) -> int:
    """Create a deterministic seed from course parameters."""
    parts = [config.wind_speed_kts, config.wind_direction_deg,
             config.start_x, config.start_y,
             config.mark_x, config.mark_y,
             config.finish_x, config.finish_y]
    key = "|".join(f"{x:.4f}" for x in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % (2**31)


class WindField:
    """Spatially-varying wind field using smooth sinusoidal perturbations."""

    def __init__(self, base_speed: float, base_direction: float,
                 seed: int = 42, n_components: int = 4):
        self.base_speed = base_speed
        self.base_direction = base_direction
        rng = np.random.RandomState(seed)

        self.speed_freqs = rng.uniform(2.0, 6.0, size=(n_components, 2))
        self.speed_phases = rng.uniform(0, 2 * np.pi, size=n_components)
        self.speed_amps = rng.uniform(0.08, 0.18, size=n_components) * base_speed

        self.dir_freqs = rng.uniform(2.0, 6.0, size=(n_components, 2))
        self.dir_phases = rng.uniform(0, 2 * np.pi, size=n_components)
        self.dir_amps = rng.uniform(3.0, 12.0, size=n_components)

    def at(self, x: float, y: float) -> tuple[float, float]:
        """Get (speed, direction) at a single point."""
        speed = self.base_speed
        for i in range(len(self.speed_amps)):
            speed += self.speed_amps[i] * np.sin(
                self.speed_freqs[i, 0] * x + self.speed_freqs[i, 1] * y + self.speed_phases[i])
        speed = max(float(speed), 3.0)

        direction = self.base_direction
        for i in range(len(self.dir_amps)):
            direction += self.dir_amps[i] * np.sin(
                self.dir_freqs[i, 0] * x + self.dir_freqs[i, 1] * y + self.dir_phases[i])
        direction = float(direction) % 360
        return speed, direction

    def at_array(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized: get wind at arrays of points."""
        speed = np.full_like(x, self.base_speed, dtype=float)
        for i in range(len(self.speed_amps)):
            speed += self.speed_amps[i] * np.sin(
                self.speed_freqs[i, 0] * x + self.speed_freqs[i, 1] * y + self.speed_phases[i])
        speed = np.maximum(speed, 3.0)

        direction = np.full_like(x, self.base_direction, dtype=float)
        for i in range(len(self.dir_amps)):
            direction += self.dir_amps[i] * np.sin(
                self.dir_freqs[i, 0] * x + self.dir_freqs[i, 1] * y + self.dir_phases[i])
        direction = direction % 360
        return speed, direction

    def get_grid(self, x_min: float, x_max: float,
                 y_min: float, y_max: float, n: int = 12) -> dict:
        """Generate a visualization grid of wind vectors."""
        xs = np.linspace(x_min, x_max, n)
        ys = np.linspace(y_min, y_max, n)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        speeds, directions = self.at_array(X.ravel(), Y.ravel())
        return {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "nx": n, "ny": n,
            "speeds": speeds.reshape(n, n).tolist(),
            "directions": directions.reshape(n, n).tolist(),
        }


# ============================================================
# Section 5: VMG Optimizer & Path Planner
# ============================================================

def point_of_sail(twa_deg: float) -> str:
    """Return the name of the point of sail for a given TWA."""
    twa = abs(twa_deg)
    if twa < 30:
        return "In Irons"
    if twa < 50:
        return "Close-Hauled"
    if twa < 70:
        return "Close Reach"
    if twa < 110:
        return "Beam Reach"
    if twa < 140:
        return "Broad Reach"
    if twa < 170:
        return "Running"
    return "Dead Run"


def normalize_angle(deg: float) -> float:
    """Normalize angle to [-180, 180)."""
    deg = deg % 360
    if deg >= 180:
        deg -= 360
    return deg


def heading_to_vector(heading_deg: float) -> np.ndarray:
    """Convert compass heading (0=N, 90=E) to unit vector (dx, dy)."""
    rad = math.radians(heading_deg)
    return np.array([math.sin(rad), math.cos(rad)])


def find_optimal_vmg(tws: float, interp: RegularGridInterpolator) -> tuple[float, float, float]:
    """Find the TWA that maximizes upwind VMG.

    Returns: (optimal_twa_deg, boat_speed_kts, vmg_kts)
    """
    twa_range = np.arange(25.0, 90.5, 0.5)
    best_twa = 45.0
    best_vmg = 0.0
    best_speed = 0.0

    for twa in twa_range:
        speed = get_boat_speed(tws, twa, interp)
        vmg = speed * math.cos(math.radians(twa))
        if vmg > best_vmg:
            best_vmg = vmg
            best_twa = float(twa)
            best_speed = speed

    return best_twa, best_speed, best_vmg


def find_optimal_downwind_vmg(
    tws: float, interp: RegularGridInterpolator
) -> tuple[float, float, float]:
    """Find the TWA that maximizes downwind VMG.

    Returns: (optimal_twa_deg, boat_speed_kts, vmg_downwind_kts)
    """
    twa_range = np.arange(90.0, 180.5, 0.5)
    best_twa = 150.0
    best_vmg = 0.0
    best_speed = 0.0

    for twa in twa_range:
        speed = get_boat_speed(tws, twa, interp)
        vmg = speed * (-math.cos(math.radians(twa)))  # downwind component
        if vmg > best_vmg:
            best_vmg = vmg
            best_twa = float(twa)
            best_speed = speed

    return best_twa, best_speed, best_vmg


# ============================================================
# Section 5b: Grid-Based Optimal Path Finder
# ============================================================

NEIGHBOR_OFFSETS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
    (2, 1), (2, -1), (-2, 1), (-2, -1),
]

GRID_SIZE = 70

# Maneuver penalties in seconds — cost of tacking or jibing per boat type
MANEUVER_PENALTIES = {
    "laser": {"tack": 6, "jibe": 5},
    "420": {"tack": 8, "jibe": 10},
    "j24": {"tack": 12, "jibe": 10},
}


def build_course_graph(
    wind_field: WindField, interp: RegularGridInterpolator,
    xs: np.ndarray, ys: np.ndarray, grid_size: int,
    boat_type: str = "laser",
) -> csr_matrix:
    """Build directed graph with tack-state-aware edges.

    State: (i, j, tack) encoded as (i * grid_size + j) * 2 + tack_idx.
    tack_idx: 0 = starboard, 1 = port.
    Tack/jibe changes incur a time penalty.
    """
    n_base = grid_size * grid_size
    n_nodes = n_base * 2
    all_rows, all_cols, all_costs = [], [], []

    penalties = MANEUVER_PENALTIES.get(boat_type, MANEUVER_PENALTIES["laser"])
    tack_pen = penalties["tack"]
    jibe_pen = penalties["jibe"]

    for di, dj in NEIGHBOR_OFFSETS:
        i_lo, i_hi = max(0, -di), min(grid_size, grid_size - di)
        j_lo, j_hi = max(0, -dj), min(grid_size, grid_size - dj)
        if i_lo >= i_hi or j_lo >= j_hi:
            continue

        ii = np.arange(i_lo, i_hi)
        jj = np.arange(j_lo, j_hi)
        I, J = np.meshgrid(ii, jj, indexing='ij')
        I_flat, J_flat = I.ravel(), J.ravel()
        NI, NJ = I_flat + di, J_flat + dj

        edge_dx = float(xs[1] - xs[0]) * di
        edge_dy = float(ys[1] - ys[0]) * dj
        edge_dist = math.sqrt(edge_dx**2 + edge_dy**2)
        heading = math.degrees(math.atan2(edge_dx, edge_dy)) % 360

        mid_x = (xs[I_flat] + xs[NI]) / 2
        mid_y = (ys[J_flat] + ys[NJ]) / 2
        ws, wd = wind_field.at_array(mid_x, mid_y)

        twa = np.abs(((heading - wd) % 360 + 180) % 360 - 180)
        ws_c = np.clip(ws, 6.0, 20.0)
        twa_c = np.minimum(twa, 180.0)
        speeds = interp(np.column_stack([ws_c, twa_c]))
        speeds = np.maximum(speeds, 0.01)

        base_time = edge_dist / (speeds * KNOTS_TO_NM_PER_SEC)

        # Determine tack for this edge: starboard(0) if heading clockwise from wind
        cross = ((heading - wd) % 360 + 360) % 360
        edge_tack = np.where((cross > 0) & (cross < 180), 0, 1).astype(int)

        # Penalty depends on whether this is a tack (upwind) or jibe (downwind)
        maneuver_cost = np.where(twa < 90, tack_pen, jibe_pen)

        base_src = I_flat * grid_size + J_flat
        base_dst = NI * grid_size + NJ

        # Create edges for both source tack states
        for src_tack in [0, 1]:
            penalty = np.where(edge_tack != src_tack, maneuver_cost, 0.0)
            total_cost = base_time + penalty

            src_ids = base_src * 2 + src_tack
            dst_ids = base_dst * 2 + edge_tack

            all_rows.append(src_ids)
            all_cols.append(dst_ids)
            all_costs.append(total_cost)

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    costs = np.concatenate(all_costs)
    return csr_matrix((costs, (rows, cols)), shape=(n_nodes, n_nodes))


def find_optimal_grid_path(
    graph: csr_matrix, start_id: int, end_id: int,
    xs: np.ndarray, ys: np.ndarray, grid_size: int,
) -> list[tuple[float, float]]:
    """Find shortest path on tack-state-extended graph.

    Tries both starting tack states, picks the fastest overall.
    """
    # Run Dijkstra from both start tacks at once
    start_nodes = [start_id * 2 + 0, start_id * 2 + 1]
    dist, pred = shortest_path(
        graph, method='D', indices=start_nodes, return_predecessors=True)

    # Find best combination of start tack × end tack
    best_cost = float('inf')
    best_src_idx = 0
    best_end_node = end_id * 2

    for src_idx in [0, 1]:
        for end_tack in [0, 1]:
            end_node = end_id * 2 + end_tack
            cost = dist[src_idx, end_node]
            if cost < best_cost:
                best_cost = cost
                best_src_idx = src_idx
                best_end_node = end_node

    # Reconstruct path
    src_node = start_nodes[best_src_idx]
    predecessors = pred[best_src_idx]
    path_ids = []
    current = best_end_node
    while current != src_node and current >= 0 and predecessors[current] != -9999:
        path_ids.append(current)
        current = int(predecessors[current])
    if current == src_node:
        path_ids.append(src_node)
    path_ids.reverse()

    # Convert extended node IDs to coordinates (strip tack state)
    return [(float(xs[(nid // 2) // grid_size]), float(ys[(nid // 2) % grid_size]))
            for nid in path_ids]


def path_to_waypoints(
    path: list[tuple[float, float]],
    wind_field: WindField, interp: RegularGridInterpolator,
    target: np.ndarray, t_offset: float, leg_name: str,
) -> list[dict]:
    """Convert a grid path to waypoints with timing and metadata."""
    waypoints = []
    t = t_offset

    for k in range(len(path)):
        x, y = path[k]
        ws, wd = wind_field.at(x, y)

        if k < len(path) - 1:
            nx_, ny_ = path[k + 1]
            edge_dx = nx_ - x
            edge_dy = ny_ - y
            edge_dist = math.sqrt(edge_dx**2 + edge_dy**2)
            heading = math.degrees(math.atan2(edge_dx, edge_dy)) % 360
        else:
            if k > 0:
                px, py = path[k - 1]
                heading = math.degrees(math.atan2(x - px, y - py)) % 360
            else:
                heading = 0.0
            edge_dist = 0.0

        twa = abs(((heading - wd) % 360 + 180) % 360 - 180)
        ws_c = max(6.0, min(20.0, ws))
        speed = max(float(interp((ws_c, min(twa, 180.0)))), 0.1)

        # VMG toward target
        target_dx = target[0] - x
        target_dy = target[1] - y
        target_dist = math.sqrt(target_dx**2 + target_dy**2)
        if target_dist > 1e-6:
            target_bearing = math.atan2(target_dx, target_dy)
            vmg = speed * math.cos(math.radians(heading) - target_bearing)
        else:
            vmg = 0.0

        cross = ((heading - wd) % 360 + 360) % 360
        tack = "starboard" if 0 < cross < 180 else "port"

        waypoints.append({
            "x": x, "y": y, "heading": heading, "tack": tack,
            "speed": speed, "vmg": abs(vmg), "time": t, "leg": leg_name,
        })

        # Advance time using midpoint wind speed (matches Dijkstra cost)
        if k < len(path) - 1:
            mx = (x + path[k + 1][0]) / 2
            my = (y + path[k + 1][1]) / 2
            mws, mwd = wind_field.at(mx, my)
            mtwa = abs(((heading - mwd) % 360 + 180) % 360 - 180)
            mws_c = max(6.0, min(20.0, mws))
            mspeed = max(float(interp((mws_c, min(mtwa, 180.0)))), 0.1)
            t += edge_dist / (mspeed * KNOTS_TO_NM_PER_SEC)

    return waypoints


def sum_path_distance(path: list[tuple[float, float]]) -> float:
    """Total distance along a path in NM."""
    total = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        total += math.sqrt(dx**2 + dy**2)
    return total


def count_tack_changes(waypoints: list[dict]) -> int:
    """Count the number of tack changes in a waypoint list."""
    changes = 0
    for i in range(1, len(waypoints)):
        if waypoints[i]["tack"] != waypoints[i - 1]["tack"]:
            changes += 1
    return changes


def compute_tack_point(
    start: np.ndarray,
    target: np.ndarray,
    heading1_deg: float,
    heading2_deg: float,
) -> tuple[np.ndarray, float, float]:
    """Find where leg1 from start meets leg2 arriving at target.

    Boat sails heading1 from start, then heading2 from tack point to target.
    Solve: start + t1*d1 + t2*d2 = target
    Returns: (tack_point, distance_leg1, distance_leg2)
    """
    d1 = heading_to_vector(heading1_deg)
    d2 = heading_to_vector(heading2_deg)

    A = np.array([[d1[0], d2[0]], [d1[1], d2[1]]])
    b = target - start

    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        return np.array([np.nan, np.nan]), -1.0, -1.0

    t = np.linalg.solve(A, b)
    tack_point = start + t[0] * d1
    return tack_point, float(t[0]), float(t[1])


def compute_leg_path(
    start: np.ndarray, target: np.ndarray,
    wind_from: float, tws: float, boat_type: str, dt: float,
    t_offset: float = 0.0, leg_name: str = "upwind",
) -> SailingPath:
    """Compute the optimal path for one leg (upwind or downwind)."""
    interp = get_interpolator(boat_type)
    diff = target - start
    target_bearing = math.degrees(math.atan2(diff[0], diff[1])) % 360
    total_straight_dist = float(np.linalg.norm(diff))

    if total_straight_dist < 1e-10:
        return SailingPath([], 0, 0, 0, 0, 0, 0, [])

    angle_off_wind = abs(normalize_angle(target_bearing - wind_from))

    # Choose upwind or downwind VMG optimization
    if angle_off_wind <= 90:
        opt_twa, opt_speed, opt_vmg = find_optimal_vmg(tws, interp)
    else:
        opt_twa, opt_speed, opt_vmg = find_optimal_downwind_vmg(tws, interp)

    starboard_heading = (wind_from + opt_twa) % 360
    port_heading = (wind_from - opt_twa) % 360

    # Compare single-leg (direct) vs two-leg (tack/jibe) paths
    direct_speed = get_boat_speed(tws, angle_off_wind, interp)
    direct_time = (
        total_straight_dist / (direct_speed * KNOTS_TO_NM_PER_SEC)
        if direct_speed > 1e-6 else float("inf")
    )

    tp_a, dist_a1, dist_a2 = compute_tack_point(
        start, target, starboard_heading, port_heading
    )
    tp_b, dist_b1, dist_b2 = compute_tack_point(
        start, target, port_heading, starboard_heading
    )

    path_a_valid = dist_a1 > 1e-6 and dist_a2 > 1e-6
    path_b_valid = dist_b1 > 1e-6 and dist_b2 > 1e-6

    time_a = (
        (dist_a1 + dist_a2) / (opt_speed * KNOTS_TO_NM_PER_SEC)
        if path_a_valid else float("inf")
    )
    time_b = (
        (dist_b1 + dist_b2) / (opt_speed * KNOTS_TO_NM_PER_SEC)
        if path_b_valid else float("inf")
    )
    best_two_time = min(time_a, time_b)

    if direct_time <= best_two_time:
        # Single leg is fastest
        cross = normalize_angle(target_bearing - wind_from)
        tack = "starboard" if cross > 0 else "port"
        actual_vmg = direct_speed * math.cos(math.radians(angle_off_wind))

        waypoints = _discretize_leg(
            start, target, target_bearing, tack,
            direct_speed, abs(actual_vmg), dt, t_offset, leg_name,
        )
        return SailingPath(
            waypoints=waypoints,
            total_distance_nm=total_straight_dist,
            total_time_seconds=(waypoints[-1].elapsed_seconds - t_offset) if waypoints else 0,
            optimal_twa_deg=opt_twa,
            optimal_vmg_kts=opt_vmg,
            optimal_speed_kts=opt_speed,
            n_tacks=0,
            legs=[{
                "tack": tack,
                "heading": target_bearing,
                "distance_nm": total_straight_dist,
                "time_s": (waypoints[-1].elapsed_seconds - t_offset) if waypoints else 0,
            }],
        )

    # Two-leg path (tack or jibe)
    if time_a <= time_b:
        tack_point = tp_a
        leg1_heading, leg2_heading = starboard_heading, port_heading
        leg1_tack, leg2_tack = "starboard", "port"
        leg1_dist, leg2_dist = dist_a1, dist_a2
    else:
        tack_point = tp_b
        leg1_heading, leg2_heading = port_heading, starboard_heading
        leg1_tack, leg2_tack = "port", "starboard"
        leg1_dist, leg2_dist = dist_b1, dist_b2

    wp_leg1 = _discretize_leg(
        start, tack_point, leg1_heading, leg1_tack,
        opt_speed, opt_vmg, dt, t_offset, leg_name,
    )
    t_after_leg1 = wp_leg1[-1].elapsed_seconds if wp_leg1 else t_offset
    wp_leg2 = _discretize_leg(
        tack_point, target, leg2_heading, leg2_tack,
        opt_speed, opt_vmg, dt, t_after_leg1, leg_name,
    )

    all_waypoints = wp_leg1 + wp_leg2
    total_time = (all_waypoints[-1].elapsed_seconds - t_offset) if all_waypoints else 0

    return SailingPath(
        waypoints=all_waypoints,
        total_distance_nm=leg1_dist + leg2_dist,
        total_time_seconds=total_time,
        optimal_twa_deg=opt_twa,
        optimal_vmg_kts=opt_vmg,
        optimal_speed_kts=opt_speed,
        n_tacks=1,
        legs=[
            {"tack": leg1_tack, "heading": leg1_heading,
             "distance_nm": leg1_dist, "time_s": t_after_leg1 - t_offset},
            {"tack": leg2_tack, "heading": leg2_heading,
             "distance_nm": leg2_dist, "time_s": total_time - (t_after_leg1 - t_offset)},
        ],
    )


def compute_full_course(config: SimConfig) -> dict:
    """Compute optimal path using grid-based Dijkstra with spatial wind."""
    start = np.array([config.start_x, config.start_y])
    mark = np.array([config.mark_x, config.mark_y])
    finish = np.array([config.finish_x, config.finish_y])
    interp = get_interpolator(config.boat_type)

    # Create spatial wind field
    seed = make_wind_seed(config)
    wind_field = WindField(config.wind_speed_kts, config.wind_direction_deg, seed=seed)

    # Grid bounds covering entire course with padding
    all_x = [config.start_x, config.mark_x, config.finish_x]
    all_y = [config.start_y, config.mark_y, config.finish_y]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    pad = max(x_max - x_min, y_max - y_min, 0.1) * 0.4
    x_min -= pad; x_max += pad
    y_min -= pad; y_max += pad

    grid_size = GRID_SIZE
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)

    # Build tack-state-aware graph (nodes doubled for starboard/port state)
    graph = build_course_graph(wind_field, interp, xs, ys, grid_size, config.boat_type)

    si = int(np.argmin(np.abs(xs - start[0])))
    sj = int(np.argmin(np.abs(ys - start[1])))
    mi = int(np.argmin(np.abs(xs - mark[0])))
    mj = int(np.argmin(np.abs(ys - mark[1])))
    fi = int(np.argmin(np.abs(xs - finish[0])))
    fj = int(np.argmin(np.abs(ys - finish[1])))

    start_id = si * grid_size + sj
    mark_id = mi * grid_size + mj
    finish_id = fi * grid_size + fj

    # Upwind: start -> mark
    upwind_path = find_optimal_grid_path(graph, start_id, mark_id, xs, ys, grid_size)

    # Downwind: mark -> finish
    downwind_path = find_optimal_grid_path(graph, mark_id, finish_id, xs, ys, grid_size)

    # Convert to waypoints
    up_wps = path_to_waypoints(upwind_path, wind_field, interp, mark, 0.0, "upwind")
    t_after_up = up_wps[-1]["time"] if up_wps else 0.0
    dn_wps = path_to_waypoints(downwind_path, wind_field, interp, finish, t_after_up, "downwind")

    all_wps = up_wps + dn_wps

    # Laylines from wind at mark
    mark_ws, mark_wd = wind_field.at(float(mark[0]), float(mark[1]))
    up_twa, _, up_vmg = find_optimal_vmg(mark_ws, interp)
    dn_twa, _, dn_vmg = find_optimal_downwind_vmg(mark_ws, interp)

    return {
        "waypoints": all_wps,
        "upwind_twa": up_twa,
        "upwind_vmg": up_vmg,
        "downwind_twa": dn_twa,
        "downwind_vmg": dn_vmg,
        "n_tacks": count_tack_changes(up_wps),
        "n_jibes": count_tack_changes(dn_wps),
        "total_distance_nm": sum_path_distance(upwind_path) + sum_path_distance(downwind_path),
        "total_time_seconds": all_wps[-1]["time"] if all_wps else 0.0,
        "laylines": {
            "upwind_sb": (mark_wd + up_twa) % 360,
            "upwind_port": (mark_wd - up_twa) % 360,
            "downwind_sb": (mark_wd + dn_twa) % 360,
            "downwind_port": (mark_wd - dn_twa) % 360,
        },
        "wind_grid": wind_field.get_grid(x_min, x_max, y_min, y_max, n=12),
        "wind_field": wind_field,
    }


def _discretize_leg(
    start: np.ndarray, end: np.ndarray,
    heading_deg: float, tack: str,
    speed_kts: float, vmg_kts: float,
    dt: float, t_offset: float,
    leg: str = "upwind",
) -> list[Waypoint]:
    """Discretize a straight-line leg into waypoints at dt intervals."""
    dist = float(np.linalg.norm(end - start))
    if dist < 1e-10:
        return []
    speed_nm_per_s = speed_kts * KNOTS_TO_NM_PER_SEC
    if speed_nm_per_s < 1e-10:
        return []

    leg_time = dist / speed_nm_per_s
    n_steps = min(max(1, int(math.ceil(leg_time / dt))), 5000)
    direction = (end - start) / dist

    waypoints = []
    for i in range(n_steps + 1):
        frac = min(i / n_steps, 1.0)
        pos = start + frac * dist * direction
        waypoints.append(Waypoint(
            x=float(pos[0]), y=float(pos[1]),
            heading_deg=heading_deg, tack=tack,
            speed_kts=speed_kts, vmg_kts=vmg_kts,
            elapsed_seconds=t_offset + frac * leg_time,
            leg=leg,
        ))
    return waypoints


def compute_user_leg(
    start: np.ndarray, via: np.ndarray, end: np.ndarray,
    wind_from: float, tws: float, boat_type: str, dt: float,
    t_offset: float = 0.0, leg_name: str = "upwind",
) -> SailingPath:
    """Compute a user-defined two-segment leg using polar speed at actual heading."""
    interp = get_interpolator(boat_type)
    all_wps: list[Waypoint] = []
    total_dist = 0.0
    t = t_offset
    n_tacks = 0

    segments = [(start, via), (via, end)]
    prev_tack = None

    for seg_start, seg_end in segments:
        diff = seg_end - seg_start
        dist = float(np.linalg.norm(diff))
        if dist < 1e-10:
            continue

        heading = math.degrees(math.atan2(diff[0], diff[1])) % 360
        twa = abs(normalize_angle(heading - wind_from))
        speed = get_boat_speed(tws, twa, interp)

        # Determine tack
        cross = normalize_angle(heading - wind_from)
        tack = "starboard" if cross > 0 else "port"
        if prev_tack is not None and tack != prev_tack:
            n_tacks += 1
        prev_tack = tack

        if speed < 0.5:
            speed = 0.5

        vmg = speed * math.cos(math.radians(twa))

        wps = _discretize_leg(
            seg_start, seg_end, heading, tack,
            speed, abs(vmg), dt, t, leg_name,
        )
        if wps:
            all_wps.extend(wps)
            t = wps[-1].elapsed_seconds
            total_dist += dist

    total_time = (all_wps[-1].elapsed_seconds - t_offset) if all_wps else 0.0

    return SailingPath(
        waypoints=all_wps,
        total_distance_nm=total_dist,
        total_time_seconds=total_time,
        optimal_twa_deg=0,
        optimal_vmg_kts=0,
        optimal_speed_kts=0,
        n_tacks=n_tacks,
        legs=[],
    )


def compute_user_leg_spatial(
    start: np.ndarray, end: np.ndarray,
    via_points: list[np.ndarray],
    wind_field: WindField, interp: RegularGridInterpolator,
    t_offset: float, leg_name: str, n_samples: int = 40,
) -> dict:
    """Compute a user-defined multi-segment leg using local wind at each sample."""
    all_wps: list[dict] = []
    total_dist = 0.0
    t = t_offset

    # Build segment list: start -> via1 -> via2 -> ... -> end
    nodes = [start] + via_points + [end]
    segments = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

    for seg_start, seg_end in segments:
        diff = seg_end - seg_start
        dist = float(np.linalg.norm(diff))
        if dist < 1e-10:
            continue

        heading = math.degrees(math.atan2(diff[0], diff[1])) % 360
        direction = diff / dist

        for i in range(n_samples + 1):
            frac = i / n_samples
            pos = seg_start + frac * dist * direction
            x, y = float(pos[0]), float(pos[1])

            ws, wd = wind_field.at(x, y)
            twa = abs(((heading - wd) % 360 + 180) % 360 - 180)
            ws_c = max(6.0, min(20.0, ws))
            speed = max(float(interp((ws_c, min(twa, 180.0)))), 0.5)

            cross = ((heading - wd) % 360 + 360) % 360
            tack = "starboard" if 0 < cross < 180 else "port"
            vmg = speed * math.cos(math.radians(twa))

            all_wps.append({
                "x": x, "y": y, "heading": heading, "tack": tack,
                "speed": speed, "vmg": abs(vmg), "time": t, "leg": leg_name,
            })

            if i < n_samples:
                seg_time = (dist / n_samples) / (speed * KNOTS_TO_NM_PER_SEC)
                t += seg_time

        total_dist += dist

    total_time = (all_wps[-1]["time"] - t_offset) if all_wps else 0.0
    return {"waypoints": all_wps, "distance": total_dist, "time": total_time}


def compute_user_course_spatial(
    config: SimConfig, wind_field: WindField,
    upwind_points: list[list[float]],
    downwind_points: list[list[float]],
) -> dict:
    """Compute user-defined course through their waypoints with spatial wind."""
    start = np.array([config.start_x, config.start_y])
    mark = np.array([config.mark_x, config.mark_y])
    finish = np.array([config.finish_x, config.finish_y])
    interp = get_interpolator(config.boat_type)

    up_via = [np.array(p) for p in upwind_points]
    dn_via = [np.array(p) for p in downwind_points]

    u_up = compute_user_leg_spatial(
        start, mark, up_via, wind_field, interp, 0.0, "upwind")
    u_t = u_up["waypoints"][-1]["time"] if u_up["waypoints"] else 0.0
    u_dn = compute_user_leg_spatial(
        mark, finish, dn_via, wind_field, interp, u_t, "downwind")

    u_all_wps = u_up["waypoints"] + u_dn["waypoints"]
    n_tacks = sum(1 for i in range(1, len(u_up["waypoints"]))
                  if u_up["waypoints"][i]["tack"] != u_up["waypoints"][i-1]["tack"])
    n_jibes = sum(1 for i in range(1, len(u_dn["waypoints"]))
                  if u_dn["waypoints"][i]["tack"] != u_dn["waypoints"][i-1]["tack"])
    return {
        "waypoints": u_all_wps,
        "total_distance_nm": u_up["distance"] + u_dn["distance"],
        "total_time_seconds": u_all_wps[-1]["time"] if u_all_wps else 0.0,
        "n_tacks": n_tacks,
        "n_jibes": n_jibes,
    }


def format_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ============================================================
# Section 6: Embedded Web App
# ============================================================

APP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sailing Simulator</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a1628;color:#ccddee;font-family:'SF Mono',ui-monospace,'Cascadia Code',monospace;
  display:flex;flex-direction:column;height:100vh;overflow:hidden}
#controls{background:#111d33;padding:12px 20px;display:flex;flex-wrap:wrap;gap:16px;
  align-items:center;border-bottom:1px solid #1a2a44}
.ctrl{display:flex;flex-direction:column;gap:2px}
.ctrl label{font-size:11px;color:#6688aa;text-transform:uppercase;letter-spacing:0.5px}
.ctrl select{background:#0d1f3c;color:#ccddee;border:1px solid #1a3355;
  border-radius:4px;padding:4px 8px;font-family:inherit;font-size:13px;cursor:pointer;min-width:80px}
.ctrl input[type=range]{width:120px;cursor:pointer;accent-color:#44aaff;background:transparent;border:none}
.ctrl .val{font-size:12px;color:#88bbdd;min-width:40px;text-align:right}
.ctrl-row{display:flex;align-items:center;gap:6px}
.ctrl input[type=checkbox]{accent-color:#44aaff;cursor:pointer;width:16px;height:16px}
.action-btn{border:none;border-radius:6px;padding:8px 24px;
  font-family:inherit;font-size:13px;cursor:pointer;font-weight:600;letter-spacing:0.5px;
  transition:background 0.15s}
#simulate-btn{background:#1a5599;color:#eef}
#simulate-btn:hover{background:#2266bb}
#simulate-btn:active{background:#0e3d77}
#reset-btn{background:#553322;color:#eef}
#reset-btn:hover{background:#774433}
#newcourse-btn{background:#224455;color:#eef}
#newcourse-btn:hover{background:#336677}
#main{flex:1;position:relative;min-height:0}
canvas{width:100%;height:100%;display:block}
#info{position:absolute;top:12px;left:12px;background:rgba(10,22,40,0.85);
  padding:10px 14px;border-radius:6px;border:1px solid #1a2a44;font-size:13px;
  line-height:1.7;pointer-events:none;min-width:180px}
#info .row{display:flex;justify-content:space-between;gap:12px}
#info .lbl{color:#6688aa}
#info .val{color:#ccddee;text-align:right}
#info .sail-val{color:#44ddff}
#info .stbd{color:#00ff88}
#info .port{color:#ff4444}
#summary{position:absolute;top:12px;right:12px;background:rgba(10,22,40,0.85);
  padding:10px 14px;border-radius:6px;border:1px solid #1a2a44;font-size:12px;
  line-height:1.6;pointer-events:none;min-width:200px}
#summary .lbl{color:#6688aa}
#summary .val{color:#ccddee}
#finish-overlay{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-size:28px;font-weight:700;color:#00ff88;pointer-events:none;opacity:0;
  text-shadow:0 0 20px rgba(0,255,136,0.4);transition:opacity 0.5s}
#challenge-hint{font-size:11px;color:#88bbdd;max-width:200px;line-height:1.4}
#score-panel{display:none;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  background:rgba(10,22,40,0.95);padding:24px 32px;border-radius:12px;border:1px solid #1a2a44;
  text-align:center;pointer-events:none;z-index:10}
#score-label{font-size:14px;color:#6688aa;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px}
#score-pct{font-size:48px;font-weight:700}
#score-delta{font-size:16px;margin-top:8px}
#score-times{font-size:12px;color:#6688aa;margin-top:12px}
</style>
</head>
<body>
<div id="controls">
  <div class="ctrl">
    <label>Mode</label>
    <select id="mode"><option value="challenge" selected>Challenge</option><option value="simulate">Simulate</option></select>
  </div>
  <div class="ctrl">
    <label>Boat</label>
    <select id="boat"><option value="laser">Laser</option><option value="420">420</option><option value="j24">J/24</option></select>
  </div>
  <div class="ctrl slider-ctrl" id="ctrl-windSpeed">
    <label>Wind Speed</label>
    <div class="ctrl-row">
      <input type="range" id="windSpeed" min="6" max="20" step="1" value="12">
      <span class="val" id="windSpeedVal">12 kts</span>
    </div>
  </div>
  <div class="ctrl slider-ctrl" id="ctrl-windDir">
    <label>Wind Direction</label>
    <div class="ctrl-row">
      <input type="range" id="windDir" min="-90" max="90" step="1" value="0">
      <span class="val" id="windDirVal">000&deg;</span>
    </div>
  </div>
  <div class="ctrl slider-ctrl" id="ctrl-startX">
    <label>Start X</label>
    <div class="ctrl-row">
      <input type="range" id="startX" min="-0.5" max="0.5" step="0.05" value="-0.15">
      <span class="val" id="startXVal">-0.15</span>
    </div>
  </div>
  <div class="ctrl slider-ctrl" id="ctrl-startY">
    <label>Start Y</label>
    <div class="ctrl-row">
      <input type="range" id="startY" min="-0.3" max="0.3" step="0.05" value="0">
      <span class="val" id="startYVal">0.00</span>
    </div>
  </div>
  <div class="ctrl slider-ctrl" id="ctrl-markX">
    <label>Mark X</label>
    <div class="ctrl-row">
      <input type="range" id="markX" min="-0.5" max="0.5" step="0.05" value="0">
      <span class="val" id="markXVal">0.00</span>
    </div>
  </div>
  <div class="ctrl slider-ctrl" id="ctrl-markY">
    <label>Mark Y</label>
    <div class="ctrl-row">
      <input type="range" id="markY" min="0.5" max="1.5" step="0.05" value="1">
      <span class="val" id="markYVal">1.00</span>
    </div>
  </div>
  <div class="ctrl slider-ctrl" id="ctrl-finishX">
    <label>Finish X</label>
    <div class="ctrl-row">
      <input type="range" id="finishX" min="-0.5" max="0.5" step="0.05" value="0.15">
      <span class="val" id="finishXVal">0.15</span>
    </div>
  </div>
  <div class="ctrl slider-ctrl" id="ctrl-finishY">
    <label>Finish Y</label>
    <div class="ctrl-row">
      <input type="range" id="finishY" min="-0.3" max="0.3" step="0.05" value="0">
      <span class="val" id="finishYVal">0.00</span>
    </div>
  </div>
  <div class="ctrl">
    <label>Laylines</label>
    <input type="checkbox" id="showLaylines">
  </div>
  <button id="simulate-btn" class="action-btn">Race</button>
  <button id="reset-btn" class="action-btn" style="display:none">Reset</button>
  <button id="newcourse-btn" class="action-btn">New Course</button>
  <button id="phase-btn" class="action-btn" style="background:#2a5544;display:none">Upwind ▲</button>
  <button id="undo-btn" class="action-btn" style="background:#443322;display:none">Undo</button>
  <div id="challenge-hint">Click to place <b>upwind (tack)</b> points.</div>
</div>
<div id="main">
  <canvas id="sim"></canvas>
  <div id="info">
    <div class="row"><span class="lbl">Leg</span><span class="val sail-val" id="i-leg">--</span></div>
    <div class="row"><span class="lbl">Sail</span><span class="val sail-val" id="i-sail">--</span></div>
    <div class="row"><span class="lbl">Speed</span><span class="val" id="i-speed">--</span></div>
    <div class="row"><span class="lbl">VMG</span><span class="val" id="i-vmg">--</span></div>
    <div class="row"><span class="lbl">Heading</span><span class="val" id="i-hdg">--</span></div>
    <div class="row"><span class="lbl">Tack</span><span class="val" id="i-tack">--</span></div>
    <div class="row"><span class="lbl">Time</span><span class="val" id="i-time">--</span></div>
  </div>
  <div id="summary" style="display:none">
    <div style="margin-bottom:4px;color:#6688aa;font-size:11px;text-transform:uppercase;letter-spacing:0.5px">Upwind</div>
    <div><span class="lbl">TWA: </span><span class="val" id="s-up-twa"></span></div>
    <div><span class="lbl">VMG: </span><span class="val" id="s-up-vmg"></span></div>
    <div><span class="lbl">Tacks: </span><span class="val" id="s-tacks"></span></div>
    <div style="margin-top:6px;margin-bottom:4px;color:#6688aa;font-size:11px;text-transform:uppercase;letter-spacing:0.5px">Downwind</div>
    <div><span class="lbl">TWA: </span><span class="val" id="s-dn-twa"></span></div>
    <div><span class="lbl">VMG: </span><span class="val" id="s-dn-vmg"></span></div>
    <div><span class="lbl">Jibes: </span><span class="val" id="s-jibes"></span></div>
    <div style="margin-top:6px;border-top:1px solid #1a2a44;padding-top:6px">
      <div><span class="lbl">Distance: </span><span class="val" id="s-dist"></span></div>
      <div><span class="lbl">Total Time: </span><span class="val" id="s-time"></span></div>
    </div>
  </div>
  <div id="finish-overlay"></div>
  <div id="score-panel">
    <div id="score-label">Tactics Score</div>
    <div id="score-pct"></div>
    <div id="score-delta"></div>
    <div id="score-times"></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);

const canvas = $('sim');
const ctx = canvas.getContext('2d');

// --- Core state ---
let waypoints = null, summary = null, animId = null;
let simState = 'idle'; // 'idle' | 'running' | 'paused' | 'finished'

// --- Challenge mode state ---
let mode = 'challenge';
let userUpwindPoints = [];    // [{x, y}, ...] world coords
let userDownwindPoints = [];  // [{x, y}, ...] world coords
let placingPhase = 'upwind';  // 'upwind' | 'downwind'
let userWaypoints = null;
let userSummary = null;
let laylineData = null;     // from server response
let windGridData = null;    // spatial wind field for visualization

// --- Animation state ---
let simElapsed = 0;       // simulate mode
let timeStep = 0;         // simulate mode
let maxTime = 0;
let optElapsed = 0;       // challenge: optimal boat sim time
let userElapsed = 0;      // challenge: user boat sim time
let optStep = 0;          // challenge: per-frame advance
let userStep = 0;
let optFinishTime = 0;
let userFinishTime = 0;
let cachedBounds = null;

const controls = ['boat','windSpeed','windDir','startX','startY','markX','markY','finishX','finishY','mode'];

function setControlsEnabled(enabled) {
  for (const id of controls) {
    const el = $(id);
    el.disabled = !enabled;
    const ctrl = el.closest('.ctrl');
    if (ctrl) ctrl.style.opacity = enabled ? '1' : '0.5';
  }
}

// --- Slider live updates + course redraw ---
for (const [id, valId, fmt] of [
  ['windSpeed','windSpeedVal', v => v+' kts'],
  ['windDir','windDirVal', v => String(((+v)%360+360)%360).padStart(3,'0')+'°'],
  ['startX','startXVal', v => parseFloat(v).toFixed(2)],
  ['startY','startYVal', v => parseFloat(v).toFixed(2)],
  ['markX','markXVal', v => parseFloat(v).toFixed(2)],
  ['markY','markYVal', v => parseFloat(v).toFixed(2)],
  ['finishX','finishXVal', v => parseFloat(v).toFixed(2)],
  ['finishY','finishYVal', v => parseFloat(v).toFixed(2)],
]) {
  $(id).addEventListener('input', () => {
    $(valId).textContent = fmt($(id).value);
    if (simState === 'idle') { userUpwindPoints = []; userDownwindPoints = []; placingPhase = 'upwind'; drawCourse(); }
  });
}
$('boat').addEventListener('change', () => {
  if (simState === 'idle') drawCourse();
});

// --- Randomize course for challenge mode ---
function randomizeCourse() {
  const windDir = Math.round((Math.random() * 120) - 60); // -60 to +60
  const startX = +(Math.random() * 0.6 - 0.3).toFixed(2);  // -0.3 to +0.3
  const startY = +(Math.random() * 0.4 - 0.2).toFixed(2);   // -0.2 to +0.2
  const markX = +(Math.random() * 0.4 - 0.2).toFixed(2);    // -0.2 to +0.2
  const markY = +(Math.random() * 0.6 + 0.7).toFixed(2);    // 0.7 to 1.3
  // Ensure finish is not at the same position as start (minimum 0.1 separation)
  let finishX, finishY;
  do {
    finishX = +(Math.random() * 0.6 - 0.3).toFixed(2);
    finishY = +(Math.random() * 0.4 - 0.2).toFixed(2);
  } while (Math.sqrt((finishX - startX)**2 + (finishY - startY)**2) < 0.1);

  $('windDir').value = windDir;
  $('windDirVal').textContent = String(((windDir % 360) + 360) % 360).padStart(3, '0') + '°';
  $('startX').value = startX;
  $('startXVal').textContent = startX.toFixed(2);
  $('startY').value = startY;
  $('startYVal').textContent = startY.toFixed(2);
  $('markX').value = markX;
  $('markXVal').textContent = markX.toFixed(2);
  $('markY').value = markY;
  $('markYVal').textContent = markY.toFixed(2);
  $('finishX').value = finishX;
  $('finishXVal').textContent = finishX.toFixed(2);
  $('finishY').value = finishY;
  $('finishYVal').textContent = finishY.toFixed(2);
}

const courseControls = ['windDir', 'startX', 'startY', 'markX', 'markY', 'finishX', 'finishY'];
function setCourseControlsLocked(locked) {
  for (const id of courseControls) {
    $(id).disabled = locked;
    const ctrl = $(id).closest('.ctrl');
    if (ctrl) ctrl.style.opacity = locked ? '0.5' : '1';
  }
}

// Hide/show all slider controls (for challenge vs simulate mode)
function setSlidersVisible(visible) {
  const sliders = document.querySelectorAll('.slider-ctrl');
  for (const el of sliders) {
    el.style.display = visible ? '' : 'none';
  }
}

// --- Mode toggle ---
$('mode').addEventListener('change', () => {
  mode = $('mode').value;
  if (mode === 'challenge') {
    btn.textContent = 'Race';
    $('challenge-hint').style.display = '';
    $('newcourse-btn').style.display = '';
    phaseBtn.style.display = '';
    undoBtn.style.display = '';
    setSlidersVisible(false);
    userUpwindPoints = [];
    userDownwindPoints = [];
    placingPhase = 'upwind';
    updatePhaseBtn();
    randomizeCourse();
    setCourseControlsLocked(true);
    fetchLaylines();
  } else {
    btn.textContent = 'Simulate';
    $('challenge-hint').style.display = 'none';
    $('newcourse-btn').style.display = 'none';
    phaseBtn.style.display = 'none';
    undoBtn.style.display = 'none';
    setSlidersVisible(true);
    userUpwindPoints = [];
    userDownwindPoints = [];
    setCourseControlsLocked(false);
  }
  if (simState !== 'idle') goIdle();
  else drawCourse();
});

// --- Layline toggle ---
$('showLaylines').addEventListener('change', () => {
  if (simState === 'idle') drawCourse();
});

// --- Read current slider values ---
function getParams() {
  return {
    startX: parseFloat($('startX').value),
    startY: parseFloat($('startY').value),
    markX: parseFloat($('markX').value),
    markY: parseFloat($('markY').value),
    finishX: parseFloat($('finishX').value),
    finishY: parseFloat($('finishY').value),
    windDir: ((parseFloat($('windDir').value) % 360) + 360) % 360,
    windSpeed: parseInt($('windSpeed').value),
  };
}

// --- Canvas setup ---
function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  canvas.width = r.width * devicePixelRatio;
  canvas.height = r.height * devicePixelRatio;
  canvas.style.width = r.width + 'px';
  canvas.style.height = r.height + 'px';
  if (waypoints && simState !== 'idle') drawFrame();
  else drawCourse();
}
window.addEventListener('resize', resize);

// --- Coordinate transforms ---
function makeBounds(pts) {
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const [x,y] of pts) {
    if (x < xMin) xMin = x; if (x > xMax) xMax = x;
    if (y < yMin) yMin = y; if (y > yMax) yMax = y;
  }
  const xMid = (xMin + xMax) / 2, yMid = (yMin + yMax) / 2;
  const halfX = (xMax - xMin) / 2, halfY = (yMax - yMin) / 2;
  const half = Math.max(halfX, halfY, 0.15);
  const pad = 1.3;
  xMin = xMid - half * pad; xMax = xMid + half * pad;
  yMin = yMid - half * pad; yMax = yMid + half * pad;
  const range = Math.max(xMax - xMin, yMax - yMin);
  const cw = canvas.width, ch = canvas.height;
  const drawSize = Math.min(cw, ch) * 0.85;
  return {xMin, yMin, range, drawSize, offsetX: (cw - drawSize) / 2, offsetY: (ch - drawSize) / 2};
}

function w2c(wx, wy, b) {
  return [
    (wx - b.xMin) / b.range * b.drawSize + b.offsetX,
    canvas.height - ((wy - b.yMin) / b.range * b.drawSize + b.offsetY)
  ];
}

function c2w(cx, cy, b) {
  return [
    (cx - b.offsetX) / b.drawSize * b.range + b.xMin,
    (canvas.height - cy - b.offsetY) / b.drawSize * b.range + b.yMin
  ];
}

// --- Canvas click handler (Challenge mode multi-waypoint placement) ---
canvas.addEventListener('click', (e) => {
  if (mode !== 'challenge' || simState !== 'idle') return;
  const p = getParams();
  const allPts = [[p.startX, p.startY], [p.markX, p.markY], [p.finishX, p.finishY]];
  for (const pt of userUpwindPoints) allPts.push([pt.x, pt.y]);
  for (const pt of userDownwindPoints) allPts.push([pt.x, pt.y]);
  const b = makeBounds(allPts);

  const rect = canvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * devicePixelRatio;
  const cy = (e.clientY - rect.top) * devicePixelRatio;
  const [wx, wy] = c2w(cx, cy, b);

  if (placingPhase === 'upwind') {
    userUpwindPoints.push({x: wx, y: wy});
    updateHint();
  } else {
    userDownwindPoints.push({x: wx, y: wy});
    updateHint();
  }
  drawCourse();
});

// --- Phase toggle button ---
const phaseBtn = $('phase-btn');
const undoBtn = $('undo-btn');

phaseBtn.addEventListener('click', () => {
  if (mode !== 'challenge' || simState !== 'idle') return;
  placingPhase = placingPhase === 'upwind' ? 'downwind' : 'upwind';
  updatePhaseBtn();
  updateHint();
  drawCourse();
});

undoBtn.addEventListener('click', () => {
  if (mode !== 'challenge' || simState !== 'idle') return;
  if (placingPhase === 'upwind' && userUpwindPoints.length > 0) {
    userUpwindPoints.pop();
  } else if (placingPhase === 'downwind' && userDownwindPoints.length > 0) {
    userDownwindPoints.pop();
  }
  updateHint();
  drawCourse();
});

// Backspace still works as keyboard shortcut for undo
document.addEventListener('keydown', (e) => {
  if (mode !== 'challenge' || simState !== 'idle') return;
  if (e.key === 'Backspace') {
    e.preventDefault();
    undoBtn.click();
  }
});

function updatePhaseBtn() {
  if (placingPhase === 'upwind') {
    phaseBtn.textContent = 'Upwind ▲';
    phaseBtn.style.background = '#2a5544';
  } else {
    phaseBtn.textContent = 'Downwind ▼';
    phaseBtn.style.background = '#554422';
  }
}

function updateHint() {
  const nUp = userUpwindPoints.length;
  const nDn = userDownwindPoints.length;
  const phase = placingPhase === 'upwind' ? 'upwind (tack)' : 'downwind (jibe)';
  let msg = 'Placing <b>' + phase + '</b> points. ';
  msg += nUp + ' upwind, ' + nDn + ' downwind.';
  if (nUp > 0 || nDn > 0) msg += ' <b>Race</b> when ready.';
  $('challenge-hint').innerHTML = msg;
}

// --- Draw wind arrow ---
function drawWindArrow(dir, speed, bounds) {
  const dpr = devicePixelRatio;
  const cw = canvas.width;
  const ch = canvas.height;
  const windRad = dir * Math.PI / 180;
  const arrowLen = 40 * dpr;
  // Position at top-right of the course area
  const cx = bounds ? (bounds.offsetX + bounds.drawSize - 30*dpr) : (cw - 55*dpr);
  const cy = bounds ? (bounds.offsetY + 40*dpr) : (ch * 0.4);

  ctx.strokeStyle = 'rgba(136,187,255,0.3)';
  ctx.lineWidth = 1.5*dpr;
  ctx.beginPath(); ctx.arc(cx, cy, arrowLen*0.65, 0, Math.PI*2); ctx.stroke();

  const blowDx = -Math.sin(windRad), blowDy = -Math.cos(windRad);
  const ax1 = cx - blowDx*arrowLen*0.45, ay1 = cy + blowDy*arrowLen*0.45;
  const ax2 = cx + blowDx*arrowLen*0.45, ay2 = cy - blowDy*arrowLen*0.45;
  ctx.strokeStyle = '#88bbff';
  ctx.lineWidth = 2.5*dpr;
  ctx.beginPath(); ctx.moveTo(ax1,ay1); ctx.lineTo(ax2,ay2); ctx.stroke();

  const angle = Math.atan2(ay2-ay1, ax2-ax1);
  const hs = 10*dpr;
  ctx.beginPath();
  ctx.moveTo(ax2, ay2);
  ctx.lineTo(ax2 - hs*Math.cos(angle-0.4), ay2 - hs*Math.sin(angle-0.4));
  ctx.moveTo(ax2, ay2);
  ctx.lineTo(ax2 - hs*Math.cos(angle+0.4), ay2 - hs*Math.sin(angle+0.4));
  ctx.stroke();

  ctx.fillStyle = '#88bbff';
  ctx.font = (11*dpr)+'px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(speed+' kts', cx, cy + arrowLen*0.95);
}

// --- Draw spatial wind field arrows ---
function drawWindField(wg, b) {
  if (!wg) return;
  const dpr = devicePixelRatio;
  const {x_min, x_max, y_min, y_max, nx, ny, speeds, directions} = wg;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const wx = x_min + (x_max - x_min) * i / (nx - 1);
      const wy = y_min + (y_max - y_min) * j / (ny - 1);
      const [cx, cy] = w2c(wx, wy, b);
      const spd = speeds[i][j];
      const dir = directions[i][j];
      const len = 18 * dpr * (spd / 15);
      const rad = dir * Math.PI / 180;
      // Downwind direction in canvas coords: (-sin(rad), +cos(rad))
      const dx = -Math.sin(rad) * len * 0.5;
      const dy = Math.cos(rad) * len * 0.5;
      const alpha = 0.45 + 0.25 * (spd / 20);
      ctx.strokeStyle = 'rgba(130,185,240,' + alpha + ')';
      ctx.lineWidth = 1.8 * dpr;
      ctx.beginPath();
      ctx.moveTo(cx - dx, cy - dy);
      ctx.lineTo(cx + dx, cy + dy);
      ctx.stroke();
      // Arrowhead at downwind end
      const angle = Math.atan2(dy, dx);
      const hs = 6 * dpr;
      ctx.beginPath();
      ctx.moveTo(cx + dx, cy + dy);
      ctx.lineTo(cx + dx - hs*Math.cos(angle-0.5), cy + dy - hs*Math.sin(angle-0.5));
      ctx.moveTo(cx + dx, cy + dy);
      ctx.lineTo(cx + dx - hs*Math.cos(angle+0.5), cy + dy - hs*Math.sin(angle+0.5));
      ctx.stroke();
    }
  }
}

// --- Draw grid ---
function drawGrid(b) {
  ctx.strokeStyle = 'rgba(68,85,102,0.15)';
  ctx.lineWidth = 1;
  const gridStep = b.range > 1.5 ? 0.5 : b.range > 0.5 ? 0.2 : 0.1;
  for (let g = Math.floor(b.yMin/gridStep)*gridStep; g <= b.yMin+b.range; g += gridStep) {
    const [x1,y1] = w2c(b.xMin, g, b);
    const [x2,y2] = w2c(b.xMin+b.range, g, b);
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  }
  for (let g = Math.floor(b.xMin/gridStep)*gridStep; g <= b.xMin+b.range; g += gridStep) {
    const [x1,y1] = w2c(g, b.yMin, b);
    const [x2,y2] = w2c(g, b.yMin+b.range, b);
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  }
}

// --- Draw all course marks ---
function drawMarks(p, b) {
  const dpr = devicePixelRatio;
  const [sx,sy] = w2c(p.startX, p.startY, b);
  ctx.fillStyle = '#00dd66';
  ctx.beginPath(); ctx.arc(sx, sy, 7*dpr, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = 'rgba(0,221,102,0.6)';
  ctx.font = (10*dpr)+'px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('START', sx, sy + 14*dpr);

  const [mx,my] = w2c(p.markX, p.markY, b);
  ctx.fillStyle = '#ff8833';
  ctx.save();
  ctx.translate(mx, my);
  ctx.rotate(Math.PI/4);
  ctx.fillRect(-6*dpr, -6*dpr, 12*dpr, 12*dpr);
  ctx.restore();
  ctx.fillStyle = 'rgba(255,136,51,0.6)';
  ctx.font = (10*dpr)+'px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('MARK', mx, my - 12*dpr);

  const [fx,fy] = w2c(p.finishX, p.finishY, b);
  ctx.fillStyle = '#ff4466';
  ctx.beginPath(); ctx.arc(fx, fy, 7*dpr, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = 'rgba(255,68,102,0.6)';
  ctx.font = (10*dpr)+'px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('FINISH', fx, fy + 14*dpr);
}

// --- Draw laylines from mark ---
function drawLaylines(b, p) {
  if (!laylineData) return;
  const dpr = devicePixelRatio;
  const [mx, my] = w2c(p.markX, p.markY, b);
  const lineLen = b.drawSize * 0.7;

  // Upwind laylines (extend backward from mark toward start area)
  for (const [key, color] of [['upwind_sb','rgba(0,255,136,0.25)'],['upwind_port','rgba(255,68,68,0.25)']]) {
    const heading = laylineData[key];
    const rad = heading * Math.PI / 180;
    const ex = mx - Math.sin(rad) * lineLen;
    const ey = my + Math.cos(rad) * lineLen;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * dpr;
    ctx.setLineDash([8*dpr, 6*dpr]);
    ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(ex, ey); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Downwind laylines from finish (extend from finish backward/upward)
  const [fxc, fyc] = w2c(p.finishX, p.finishY, b);
  for (const [key, color] of [['downwind_sb','rgba(0,255,136,0.15)'],['downwind_port','rgba(255,68,68,0.15)']]) {
    const heading = laylineData[key];
    const rad = heading * Math.PI / 180;
    const ex = fxc + Math.sin(rad) * lineLen;
    const ey = fyc - Math.cos(rad) * lineLen;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5 * dpr;
    ctx.setLineDash([6*dpr, 5*dpr]);
    ctx.beginPath(); ctx.moveTo(fxc, fyc); ctx.lineTo(ex, ey); ctx.stroke();
    ctx.setLineDash([]);
  }
}

// --- Draw user-placed waypoints ---
function drawUserPoints(b) {
  const dpr = devicePixelRatio;
  for (let i = 0; i < userUpwindPoints.length; i++) {
    const pt = userUpwindPoints[i];
    const [tx,ty] = w2c(pt.x, pt.y, b);
    ctx.strokeStyle = '#44ddff';
    ctx.lineWidth = 2*dpr;
    const s = 8*dpr;
    ctx.beginPath(); ctx.moveTo(tx-s,ty); ctx.lineTo(tx+s,ty); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(tx,ty-s); ctx.lineTo(tx,ty+s); ctx.stroke();
    ctx.fillStyle = '#44ddff';
    ctx.font = (10*dpr)+'px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('T' + (i+1), tx, ty - 12*dpr);
  }
  for (let i = 0; i < userDownwindPoints.length; i++) {
    const pt = userDownwindPoints[i];
    const [jx,jy] = w2c(pt.x, pt.y, b);
    ctx.strokeStyle = '#ffaa44';
    ctx.lineWidth = 2*dpr;
    const s = 8*dpr;
    ctx.beginPath(); ctx.moveTo(jx-s,jy); ctx.lineTo(jx+s,jy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(jx,jy-s); ctx.lineTo(jx,jy+s); ctx.stroke();
    ctx.fillStyle = '#ffaa44';
    ctx.font = (10*dpr)+'px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('J' + (i+1), jx, jy - 12*dpr);
  }
}

// --- Draw proposed user path (dashed lines through their points) ---
function drawUserProposedPath(b) {
  const dpr = devicePixelRatio;
  const p = getParams();
  ctx.setLineDash([4*dpr, 4*dpr]);
  ctx.lineWidth = 1.5*dpr;

  if (userUpwindPoints.length > 0) {
    // start -> tack1 -> tack2 -> ... -> mark
    ctx.strokeStyle = 'rgba(68,221,255,0.35)';
    ctx.beginPath();
    const [sx,sy] = w2c(p.startX, p.startY, b);
    ctx.moveTo(sx,sy);
    for (const pt of userUpwindPoints) {
      const [tx,ty] = w2c(pt.x, pt.y, b);
      ctx.lineTo(tx,ty);
    }
    const [mx,my] = w2c(p.markX, p.markY, b);
    ctx.lineTo(mx,my);
    ctx.stroke();
  }
  if (userDownwindPoints.length > 0) {
    // mark -> jibe1 -> jibe2 -> ... -> finish
    ctx.strokeStyle = 'rgba(255,170,68,0.35)';
    ctx.beginPath();
    const [mx,my] = w2c(p.markX, p.markY, b);
    ctx.moveTo(mx,my);
    for (const pt of userDownwindPoints) {
      const [jx,jy] = w2c(pt.x, pt.y, b);
      ctx.lineTo(jx,jy);
    }
    const [fx,fy] = w2c(p.finishX, p.finishY, b);
    ctx.lineTo(fx,fy);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

// --- Course preview (no simulation, just marks + wind) ---
function drawCourse() {
  if (animId) return;
  const p = getParams();
  const cw = canvas.width, ch = canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#0d1f3c';
  ctx.fillRect(0, 0, cw, ch);

  const allPts = [[p.startX, p.startY], [p.markX, p.markY], [p.finishX, p.finishY]];
  for (const pt of userUpwindPoints) allPts.push([pt.x, pt.y]);
  for (const pt of userDownwindPoints) allPts.push([pt.x, pt.y]);
  const b = makeBounds(allPts);
  drawGrid(b);
  drawWindField(windGridData, b);
  if ($('showLaylines').checked && laylineData) drawLaylines(b, p);
  if (mode === 'challenge') {
    drawUserProposedPath(b);
    drawUserPoints(b);
  }
  drawMarks(p, b);
  // Wind compass removed — spatial wind field arrows show local wind
}

// --- Interpolate waypoint at a given elapsed time ---
function interpAtTime(wps, t) {
  if (!wps || wps.length === 0) return null;
  if (t <= wps[0].time) return wps[0];
  if (t >= wps[wps.length - 1].time) return wps[wps.length - 1];
  // Binary search for bracket
  let lo = 0, hi = wps.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (wps[mid].time <= t) lo = mid; else hi = mid;
  }
  const w0 = wps[lo], w1 = wps[hi];
  const dt = w1.time - w0.time;
  const f = dt > 0 ? (t - w0.time) / dt : 0;
  return {
    x: w0.x + (w1.x - w0.x) * f,
    y: w0.y + (w1.y - w0.y) * f,
    heading: w1.heading,
    speed: w0.speed + (w1.speed - w0.speed) * f,
    vmg: w0.vmg + (w1.vmg - w0.vmg) * f,
    tack: w1.tack,
    leg: w1.leg,
    time: t,
  };
}
function trailIdxAtTime(wps, t) {
  if (!wps || wps.length === 0) return 0;
  for (let i = wps.length - 1; i >= 0; i--) {
    if (wps[i].time <= t) return i;
  }
  return 0;
}

// --- Draw a boat triangle ---
function drawBoat(wx, wy, headingDeg, b, color, alpha) {
  const dpr = devicePixelRatio;
  const [bx,by] = w2c(wx, wy, b);
  const headingRad = headingDeg * Math.PI / 180;
  const bs = 10*dpr;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(bx, by);
  ctx.rotate(headingRad);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(0, -bs);
  ctx.lineTo(-bs*0.6, bs*0.5);
  ctx.lineTo(bs*0.6, bs*0.5);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// --- Draw a path trail up to a given index ---
function drawTrail(wps, upToIdx, b, color, dash) {
  const dpr = devicePixelRatio;
  if (!wps || upToIdx < 1) return;
  ctx.setLineDash(dash.map(d => d*dpr));
  ctx.strokeStyle = color;
  ctx.lineWidth = 2*dpr;
  ctx.beginPath();
  for (let i = 0; i <= Math.min(upToIdx, wps.length-1); i++) {
    const [px,py] = w2c(wps[i].x, wps[i].y, b);
    i === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

// --- Full animation scene ---
function drawFrame() {
  const dpr = devicePixelRatio;
  const cw = canvas.width, ch = canvas.height;
  const p = getParams();
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#0d1f3c';
  ctx.fillRect(0, 0, cw, ch);

  // Build bounds from all waypoints
  const pts = waypoints.map(w => [w.x, w.y]);
  if (userWaypoints) {
    for (const uw of userWaypoints) pts.push([uw.x, uw.y]);
  }
  pts.push([p.startX, p.startY], [p.markX, p.markY], [p.finishX, p.finishY]);
  const b = makeBounds(pts);
  cachedBounds = b;

  drawGrid(b);
  drawWindField(windGridData, b);
  if ($('showLaylines').checked && laylineData) drawLaylines(b, p);

  const isChallenge = mode === 'challenge' && userWaypoints;

  // Optimal path (dashed preview line)
  ctx.setLineDash([6*dpr, 4*dpr]);
  ctx.strokeStyle = isChallenge ? 'rgba(68,85,102,0.3)' : 'rgba(68,85,102,0.5)';
  ctx.lineWidth = 1.5*dpr;
  ctx.beginPath();
  for (let i = 0; i < waypoints.length; i++) {
    const [px,py] = w2c(waypoints[i].x, waypoints[i].y, b);
    i === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // User planned path (dashed)
  if (isChallenge) {
    ctx.setLineDash([4*dpr, 4*dpr]);
    ctx.strokeStyle = 'rgba(68,221,255,0.2)';
    ctx.lineWidth = 1.5*dpr;
    ctx.beginPath();
    for (let i = 0; i < userWaypoints.length; i++) {
      const [px,py] = w2c(userWaypoints[i].x, userWaypoints[i].y, b);
      i === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Tack/jibe annotations on optimal path
  for (let i = 1; i < waypoints.length; i++) {
    if (waypoints[i].leg !== waypoints[i-1].leg) {
      const [tx,ty] = w2c(waypoints[i].x, waypoints[i].y, b);
      if (!isChallenge) {
        ctx.fillStyle = '#ff8833';
        ctx.font = (10*dpr)+'px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('ROUNDING', tx, ty - 14*dpr);
      }
    } else if (waypoints[i].tack !== waypoints[i-1].tack && !isChallenge) {
      const [tx,ty] = w2c(waypoints[i].x, waypoints[i].y, b);
      const isDownwind = waypoints[i].leg === 'downwind';
      ctx.strokeStyle = '#ffdd44';
      ctx.lineWidth = 2*dpr;
      const s = 6*dpr;
      ctx.beginPath(); ctx.moveTo(tx-s,ty-s); ctx.lineTo(tx+s,ty+s); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx+s,ty-s); ctx.lineTo(tx-s,ty+s); ctx.stroke();
      ctx.fillStyle = '#ffdd44';
      ctx.font = (10*dpr)+'px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(isDownwind ? 'JIBE' : 'TACK', tx, ty - 10*dpr);
    }
  }

  drawMarks(p, b);
  // Wind compass removed — spatial wind field arrows show local wind

  // Interpolate current positions (challenge uses independent clocks)
  const optT = isChallenge ? optElapsed : simElapsed;
  const userT = isChallenge ? userElapsed : simElapsed;
  const optWp = interpAtTime(waypoints, optT);
  const optTrailIdx = trailIdxAtTime(waypoints, optT);

  if (isChallenge) {
    const userWp = interpAtTime(userWaypoints, userT);
    const userTrailIdx = trailIdxAtTime(userWaypoints, userT);

    // Draw trails
    drawTrail(waypoints, optTrailIdx, b, 'rgba(150,180,220,0.3)', [4,3]);
    drawTrail(userWaypoints, userTrailIdx, b, 'rgba(68,221,255,0.4)', [0]);

    // Ghost boat (optimal) - semi-transparent
    drawBoat(optWp.x, optWp.y, optWp.heading, b, '#8899bb', 0.35);

    // User boat - solid
    drawBoat(userWp.x, userWp.y, userWp.heading, b, '#44ddff', 1.0);

    updateInfo(userWp);
  } else {
    // Single boat (simulate mode)
    drawBoat(optWp.x, optWp.y, optWp.heading, b, 'white', 1.0);
    updateInfo(optWp);
  }
}

// --- Info panel ---
function updateInfo(wp) {
  const wd = getParams().windDir;
  const twa = Math.abs(((wp.heading - wd) % 360 + 540) % 360 - 180);
  $('i-leg').textContent = wp.leg === 'downwind' ? 'Downwind' : 'Upwind';
  $('i-sail').textContent = pointOfSail(twa);
  $('i-speed').textContent = wp.speed.toFixed(1) + ' kts';
  $('i-vmg').textContent = wp.vmg.toFixed(1) + ' kts';
  $('i-hdg').textContent = String(Math.round(wp.heading)).padStart(3,'0') + '°';
  const tackEl = $('i-tack');
  tackEl.textContent = wp.tack.toUpperCase();
  tackEl.className = 'val ' + (wp.tack === 'starboard' ? 'stbd' : 'port');
  $('i-time').textContent = fmtTime(wp.time);
}

function pointOfSail(twa) {
  if (twa < 30) return 'In Irons';
  if (twa < 50) return 'Close-Hauled';
  if (twa < 70) return 'Close Reach';
  if (twa < 110) return 'Beam Reach';
  if (twa < 140) return 'Broad Reach';
  if (twa < 170) return 'Running';
  return 'Dead Run';
}

function fmtTime(s) {
  s = Math.round(s);
  const m = Math.floor(s/60), sec = s%60;
  return m + ':' + String(sec).padStart(2,'0');
}

// --- Buttons: Simulate/Race/Pause/Resume + Reset ---
const btn = $('simulate-btn');
const resetBtn = $('reset-btn');
const newCourseBtn = $('newcourse-btn');
btn.addEventListener('click', handleBtn);
resetBtn.addEventListener('click', goIdle);
newCourseBtn.addEventListener('click', () => {
  if (simState !== 'idle') goIdle();
  userUpwindPoints = [];
  userDownwindPoints = [];
  placingPhase = 'upwind';
  updateHint();
  randomizeCourse();
  fetchLaylines();
  drawCourse();
});

function stopAnim() {
  if (animId) { cancelAnimationFrame(animId); animId = null; }
}

function goIdle() {
  stopAnim();
  simState = 'idle';
  waypoints = null;
  summary = null;
  userWaypoints = null;
  userSummary = null;
  btn.textContent = mode === 'challenge' ? 'Race' : 'Simulate';
  resetBtn.style.display = 'none';
  setControlsEnabled(true);
  $('finish-overlay').style.opacity = '0';
  $('score-panel').style.display = 'none';
  $('summary').style.display = 'none';
  if (mode === 'challenge') {
    userUpwindPoints = [];
    userDownwindPoints = [];
    placingPhase = 'upwind';
    updatePhaseBtn();
    updateHint();
    $('challenge-hint').style.display = '';
    $('newcourse-btn').style.display = '';
    phaseBtn.style.display = '';
    undoBtn.style.display = '';
    setSlidersVisible(false);
    randomizeCourse();
    setCourseControlsLocked(true);
    fetchLaylines();
  } else {
    $('newcourse-btn').style.display = 'none';
    phaseBtn.style.display = 'none';
    undoBtn.style.display = 'none';
    setSlidersVisible(true);
  }
  drawCourse();
}

function handleBtn() {
  if (simState === 'idle') {
    if (mode === 'challenge' && userUpwindPoints.length === 0 && userDownwindPoints.length === 0) {
      $('challenge-hint').innerHTML = '<span style="color:#ff6666">Place at least one waypoint first!</span>';
      return;
    }
    fetchAndRun();
  } else if (simState === 'running') {
    stopAnim();
    simState = 'paused';
    btn.textContent = 'Resume';
  } else if (simState === 'paused') {
    simState = 'running';
    btn.textContent = 'Pause';
    tickLoop();
  } else if (simState === 'finished') {
    simState = 'running';
    btn.textContent = 'Pause';
    simElapsed = 0;
    optElapsed = 0;
    userElapsed = 0;
    $('finish-overlay').style.opacity = '0';
    $('score-panel').style.display = 'none';
    tickLoop();
  }
}

async function fetchAndRun() {
  btn.textContent = '...';
  btn.disabled = true;
  $('finish-overlay').style.opacity = '0';
  $('score-panel').style.display = 'none';
  setControlsEnabled(false);
  const p = getParams();
  const body = {
    boat_type: $('boat').value,
    wind_speed: p.windSpeed,
    wind_direction: p.windDir,
    start_x: p.startX,
    start_y: p.startY,
    mark_x: p.markX,
    mark_y: p.markY,
    finish_x: p.finishX,
    finish_y: p.finishY,
  };

  if (mode === 'challenge' && (userUpwindPoints.length > 0 || userDownwindPoints.length > 0)) {
    body.user_upwind_points = userUpwindPoints.map(p => [p.x, p.y]);
    body.user_downwind_points = userDownwindPoints.map(p => [p.x, p.y]);
  }

  try {
    const resp = await fetch('/compute', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    waypoints = data.waypoints;
    summary = data.summary;
    laylineData = summary.laylines || null;
    windGridData = data.wind_grid || null;

    if (data.user_waypoints) {
      userWaypoints = data.user_waypoints;
      userSummary = data.user_summary;
    } else {
      userWaypoints = null;
      userSummary = null;
    }

    // Populate summary panel (simulate mode or always)
    $('summary').style.display = 'block';
    $('s-up-twa').textContent = summary.upwind_twa.toFixed(1) + '°';
    $('s-up-vmg').textContent = summary.upwind_vmg.toFixed(1) + ' kts';
    $('s-tacks').textContent = summary.n_tacks;
    $('s-dn-twa').textContent = summary.downwind_twa.toFixed(1) + '°';
    $('s-dn-vmg').textContent = summary.downwind_vmg.toFixed(1) + ' kts';
    $('s-jibes').textContent = summary.n_jibes;
    $('s-dist').textContent = summary.total_distance_nm.toFixed(3) + ' NM';
    $('s-time').textContent = fmtTime(summary.total_time_s);

    btn.disabled = false;
    resetBtn.style.display = '';
    startAnimation();
  } catch(e) {
    console.error(e);
    btn.disabled = false;
    goIdle();
  }
}

function startAnimation() {
  stopAnim();
  simElapsed = 0;
  simState = 'running';

  optFinishTime = waypoints.length > 0 ? waypoints[waypoints.length-1].time : 0;
  userFinishTime = userWaypoints && userWaypoints.length > 0 ? userWaypoints[userWaypoints.length-1].time : 0;
  maxTime = Math.max(optFinishTime, userFinishTime);

  if (mode === 'challenge' && userFinishTime > 0) {
    // Compressed time tracks: faster boat 20s wall, slower capped at 1.3x
    const fasterTime = Math.min(optFinishTime, userFinishTime);
    const slowerTime = maxTime;
    const ratio = slowerTime / fasterTime;
    const cappedRatio = Math.min(ratio, 1.3);
    const fasterWall = 20 * 60;
    const slowerWall = fasterWall * cappedRatio;

    if (optFinishTime <= userFinishTime) {
      optStep = optFinishTime / fasterWall;
      userStep = userFinishTime / slowerWall;
    } else {
      userStep = userFinishTime / fasterWall;
      optStep = optFinishTime / slowerWall;
    }
    optElapsed = 0;
    userElapsed = 0;
  } else {
    timeStep = maxTime / (20 * 60);
  }

  btn.textContent = 'Pause';
  tickLoop();
}

function tickLoop() {
  function tick() {
    drawFrame();

    if (mode === 'challenge' && userWaypoints) {
      const optDone = optElapsed >= optFinishTime;
      const userDone = userElapsed >= userFinishTime;
      if (!optDone) optElapsed = Math.min(optElapsed + optStep, optFinishTime);
      if (!userDone) userElapsed = Math.min(userElapsed + userStep, userFinishTime);

      if (!optDone || !userDone) {
        animId = requestAnimationFrame(tick);
      } else {
        drawFrame();
        animId = null;
        simState = 'finished';
        btn.textContent = 'Replay';
        showChallengeResults();
      }
    } else {
      if (simElapsed < maxTime) {
        simElapsed += timeStep;
        animId = requestAnimationFrame(tick);
      } else {
        simElapsed = maxTime;
        drawFrame();
        animId = null;
        simState = 'finished';
        btn.textContent = 'Replay';
        $('finish-overlay').textContent = 'FINISHED  ' + fmtTime(summary.total_time_s);
        $('finish-overlay').style.opacity = '1';
      }
    }
  }
  animId = requestAnimationFrame(tick);
}

function showChallengeResults() {
  const optTime = summary.total_time_s;
  const userTime = userSummary.total_time_s;
  const score = Math.min(100, (optTime / userTime) * 100);
  const delta = userTime - optTime;

  $('score-pct').textContent = score.toFixed(1) + '%';
  if (score >= 95) $('score-pct').style.color = '#00ff88';
  else if (score >= 80) $('score-pct').style.color = '#ffdd44';
  else $('score-pct').style.color = '#ff4444';

  if (delta <= 0.5) {
    $('score-delta').textContent = 'Perfect!';
    $('score-delta').style.color = '#00ff88';
  } else {
    $('score-delta').textContent = '+' + Math.round(delta) + 's slower';
    $('score-delta').style.color = '#ff8866';
  }

  $('score-times').textContent = 'Optimal: ' + fmtTime(optTime) + '  |  Yours: ' + fmtTime(userTime);
  $('score-panel').style.display = '';
}

// --- Fetch laylines on settings change for preview ---
let laylineFetchTimer = null;
function fetchLaylines() {
  clearTimeout(laylineFetchTimer);
  laylineFetchTimer = setTimeout(async () => {
    const p = getParams();
    try {
      const resp = await fetch('/compute', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          boat_type: $('boat').value,
          wind_speed: p.windSpeed,
          wind_direction: p.windDir,
          start_x: p.startX,
          start_y: p.startY,
          mark_x: p.markX,
          mark_y: p.markY,
          finish_x: p.finishX,
          finish_y: p.finishY,
        }),
      });
      const data = await resp.json();
      laylineData = data.summary.laylines || null;
      windGridData = data.wind_grid || null;
      if (simState === 'idle') drawCourse();
    } catch(e) {}
  }, 200);
}

// Fetch laylines whenever relevant controls change
for (const id of ['windSpeed','windDir','startX','startY','markX','markY','finishX','finishY']) {
  $(id).addEventListener('input', fetchLaylines);
}
$('boat').addEventListener('change', fetchLaylines);

// Initial challenge mode setup (challenge is default)
randomizeCourse();
setCourseControlsLocked(true);
setSlidersVisible(false);
$('newcourse-btn').style.display = '';
phaseBtn.style.display = '';
undoBtn.style.display = '';
updatePhaseBtn();

// Initial fetch
fetchLaylines();

// Initial draw
resize();
</script>
</body>
</html>"""


# ============================================================
# Section 7: HTTP Server
# ============================================================

class SimHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(APP_HTML.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/compute":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            config = SimConfig(
                boat_type=body.get("boat_type", "laser"),
                wind_speed_kts=body.get("wind_speed", 12.0),
                wind_direction_deg=body.get("wind_direction", 0.0),
                start_x=body.get("start_x", 0.0),
                start_y=body.get("start_y", 0.0),
                mark_x=body.get("mark_x", 0.0),
                mark_y=body.get("mark_y", 1.0),
                finish_x=body.get("finish_x", 0.0),
                finish_y=body.get("finish_y", 0.0),
            )
            course = compute_full_course(config)
            wind_field = course["wind_field"]

            result = {
                "waypoints": course["waypoints"],
                "summary": {
                    "total_distance_nm": course["total_distance_nm"],
                    "total_time_s": course["total_time_seconds"],
                    "upwind_twa": course["upwind_twa"],
                    "upwind_vmg": course["upwind_vmg"],
                    "n_tacks": course["n_tacks"],
                    "downwind_twa": course["downwind_twa"],
                    "downwind_vmg": course["downwind_vmg"],
                    "n_jibes": course["n_jibes"],
                    "laylines": course["laylines"],
                },
                "wind_grid": course["wind_grid"],
            }

            # If user waypoints provided, compute user path
            if "user_upwind_points" in body or "user_downwind_points" in body:
                upwind_pts = body.get("user_upwind_points", [])
                downwind_pts = body.get("user_downwind_points", [])
                user_course = compute_user_course_spatial(
                    config, wind_field, upwind_pts, downwind_pts,
                )
                result["user_waypoints"] = user_course["waypoints"]
                result["user_summary"] = {
                    "total_distance_nm": user_course["total_distance_nm"],
                    "total_time_s": user_course["total_time_seconds"],
                    "n_tacks": user_course["n_tacks"],
                    "n_jibes": user_course["n_jibes"],
                }

            # Remove non-serializable wind_field before JSON encoding
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def main():
    port = 8420
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == "--port" and i < len(sys.argv) - 1:
                port = int(sys.argv[i + 1])

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), SimHandler) as httpd:
        url = f"http://localhost:{port}"
        print(f"Sailing Simulator running at {url}")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
