import json
import math
import hashlib
from http.server import BaseHTTPRequestHandler

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# ============================================================
# Constants
# ============================================================

KNOTS_TO_NM_PER_SEC = 1.0 / 3600.0

# Maneuver penalties in seconds — cost of tacking or jibing per boat type
MANEUVER_PENALTIES = {
    "laser": {"tack": 6.0, "jibe": 5.0},
    "420":   {"tack": 8.0, "jibe": 10.0},
    "j24":   {"tack": 12.0, "jibe": 10.0},
}

# ============================================================
# Polar Speed Data
# ============================================================

LASER_POLAR_DATA = {
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

FOUR_TWENTY_POLAR_DATA = {
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

J24_POLAR_DATA = {
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

BOAT_POLARS = {
    "laser": LASER_POLAR_DATA,
    "420": FOUR_TWENTY_POLAR_DATA,
    "j24": J24_POLAR_DATA,
}

_interp_cache = {}


def build_polar_interpolator(polar_data):
    tws_values = sorted(polar_data.keys())
    twa_values = [entry[0] for entry in polar_data[tws_values[0]]]
    speeds = np.array(
        [[spd for _, spd in polar_data[tws]] for tws in tws_values]
    )
    return RegularGridInterpolator(
        (np.array(tws_values, dtype=float), np.array(twa_values, dtype=float)),
        speeds, method="linear", bounds_error=False, fill_value=0.0,
    )


def get_interpolator(boat_type):
    if boat_type not in _interp_cache:
        _interp_cache[boat_type] = build_polar_interpolator(BOAT_POLARS[boat_type])
    return _interp_cache[boat_type]


def get_boat_speed(tws, twa_deg, interp):
    twa_abs = abs(twa_deg) % 360
    if twa_abs > 180:
        twa_abs = 360 - twa_abs
    tws_clamped = np.clip(tws, 6.0, 20.0)
    return float(interp((tws_clamped, twa_abs)))


# ============================================================
# Helpers
# ============================================================

def normalize_angle(deg):
    deg = deg % 360
    if deg >= 180:
        deg -= 360
    return deg


def heading_to_vector(heading_deg):
    rad = math.radians(heading_deg)
    return np.array([math.sin(rad), math.cos(rad)])


def find_optimal_vmg(tws, interp):
    twa_range = np.arange(25.0, 90.5, 0.5)
    best_twa, best_vmg, best_speed = 45.0, 0.0, 0.0
    for twa in twa_range:
        speed = get_boat_speed(tws, twa, interp)
        vmg = speed * math.cos(math.radians(twa))
        if vmg > best_vmg:
            best_vmg, best_twa, best_speed = vmg, float(twa), speed
    return best_twa, best_speed, best_vmg


def find_optimal_downwind_vmg(tws, interp):
    twa_range = np.arange(90.0, 180.5, 0.5)
    best_twa, best_vmg, best_speed = 150.0, 0.0, 0.0
    for twa in twa_range:
        speed = get_boat_speed(tws, twa, interp)
        vmg = speed * (-math.cos(math.radians(twa)))
        if vmg > best_vmg:
            best_vmg, best_twa, best_speed = vmg, float(twa), speed
    return best_twa, best_speed, best_vmg


# ============================================================
# Wind Field (spatially varying)
# ============================================================

def make_wind_seed(body):
    """Deterministic seed from course parameters."""
    parts = [
        body.get("wind_speed", 12.0),
        body.get("wind_direction", 0.0),
        body.get("start_x", 0.0),
        body.get("start_y", 0.0),
        body.get("mark_x", 0.0),
        body.get("mark_y", 1.0),
        body.get("finish_x", 0.0),
        body.get("finish_y", 0.0),
    ]
    key = "|".join(f"{x:.4f}" for x in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % (2**31)


class WindField:
    """Spatially-varying wind field using smooth sinusoidal perturbations."""

    def __init__(self, base_speed, base_direction, seed=42, n_components=4):
        self.base_speed = float(base_speed)
        self.base_direction = float(base_direction)
        self.n_components = n_components

        rng = np.random.RandomState(seed)

        # Frequency range for spatial variation (cycles per NM)
        self.speed_freq_x = rng.uniform(2.0, 6.0, n_components)
        self.speed_freq_y = rng.uniform(2.0, 6.0, n_components)
        self.speed_phase_x = rng.uniform(0.0, 2.0 * np.pi, n_components)
        self.speed_phase_y = rng.uniform(0.0, 2.0 * np.pi, n_components)
        # Speed amplitude: +/-15-20% of base_speed, distributed across components
        max_speed_perturb = rng.uniform(0.15, 0.20) * self.base_speed
        self.speed_amplitudes = rng.uniform(0.3, 1.0, n_components)
        self.speed_amplitudes *= max_speed_perturb / self.speed_amplitudes.sum()

        self.dir_freq_x = rng.uniform(2.0, 6.0, n_components)
        self.dir_freq_y = rng.uniform(2.0, 6.0, n_components)
        self.dir_phase_x = rng.uniform(0.0, 2.0 * np.pi, n_components)
        self.dir_phase_y = rng.uniform(0.0, 2.0 * np.pi, n_components)
        # Direction amplitude: +/-8-15 degrees total, distributed across components
        max_dir_perturb = rng.uniform(8.0, 15.0)
        self.dir_amplitudes = rng.uniform(0.3, 1.0, n_components)
        self.dir_amplitudes *= max_dir_perturb / self.dir_amplitudes.sum()

    def at(self, x, y):
        """Return (speed, direction) at a single point."""
        speed_perturb = 0.0
        dir_perturb = 0.0
        for k in range(self.n_components):
            speed_perturb += self.speed_amplitudes[k] * (
                math.sin(self.speed_freq_x[k] * x + self.speed_phase_x[k]) *
                math.sin(self.speed_freq_y[k] * y + self.speed_phase_y[k])
            )
            dir_perturb += self.dir_amplitudes[k] * (
                math.sin(self.dir_freq_x[k] * x + self.dir_phase_x[k]) *
                math.sin(self.dir_freq_y[k] * y + self.dir_phase_y[k])
            )
        speed = max(3.0, self.base_speed + speed_perturb)
        direction = (self.base_direction + dir_perturb) % 360.0
        return speed, direction

    def at_array(self, x_arr, y_arr):
        """Vectorized version for arrays of points. Returns (speeds, directions)."""
        x_arr = np.asarray(x_arr, dtype=float)
        y_arr = np.asarray(y_arr, dtype=float)
        speed_perturb = np.zeros_like(x_arr)
        dir_perturb = np.zeros_like(x_arr)
        for k in range(self.n_components):
            speed_perturb += self.speed_amplitudes[k] * (
                np.sin(self.speed_freq_x[k] * x_arr + self.speed_phase_x[k]) *
                np.sin(self.speed_freq_y[k] * y_arr + self.speed_phase_y[k])
            )
            dir_perturb += self.dir_amplitudes[k] * (
                np.sin(self.dir_freq_x[k] * x_arr + self.dir_phase_x[k]) *
                np.sin(self.dir_freq_y[k] * y_arr + self.dir_phase_y[k])
            )
        speeds = np.maximum(3.0, self.base_speed + speed_perturb)
        directions = (self.base_direction + dir_perturb) % 360.0
        return speeds, directions

    def get_grid(self, x_min, x_max, y_min, y_max, n=12):
        """Generate a visualization grid of wind data."""
        xs = np.linspace(x_min, x_max, n)
        ys = np.linspace(y_min, y_max, n)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
        speeds, directions = self.at_array(grid_x.ravel(), grid_y.ravel())
        speeds_2d = speeds.reshape(n, n)
        directions_2d = directions.reshape(n, n)
        return {
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
            "nx": n,
            "ny": n,
            "speeds": speeds_2d.tolist(),
            "directions": directions_2d.tolist(),
        }


# ============================================================
# Grid-based Dijkstra Pathfinder
# ============================================================

NEIGHBOR_OFFSETS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),          # cardinal
    (1, 1), (1, -1), (-1, 1), (-1, -1),         # diagonal
    (1, 2), (1, -2), (-1, 2), (-1, -2),         # knight moves
    (2, 1), (2, -1), (-2, 1), (-2, -1),
]

GRID_SIZE = 70


def build_grid_and_graph(course_points, wind_field, interp, boat_type="laser"):
    """
    Build a uniform grid covering all course points with padding,
    then construct a sparse directed graph with edge costs = sailing time.

    State: (i, j, tack) encoded as (i * grid_size + j) * 2 + tack_idx.
    tack_idx: 0 = starboard, 1 = port.
    Tack/jibe changes incur a time penalty.

    Parameters
    ----------
    course_points : list of (x, y) tuples
    wind_field : WindField
    interp : RegularGridInterpolator for polar speeds
    boat_type : str

    Returns
    -------
    xs : 1D array of x coordinates
    ys : 1D array of y coordinates
    graph : csr_matrix (N x N) with edge costs in seconds
    """
    pts = np.array(course_points, dtype=float)
    x_min_pt, y_min_pt = pts.min(axis=0)
    x_max_pt, y_max_pt = pts.max(axis=0)

    x_extent = x_max_pt - x_min_pt
    y_extent = y_max_pt - y_min_pt
    extent = max(x_extent, y_extent, 0.01)  # avoid zero extent

    pad = 0.4 * extent
    x_min = x_min_pt - pad
    x_max = x_max_pt + pad
    y_min = y_min_pt - pad
    y_max = y_max_pt + pad

    n = GRID_SIZE
    xs = np.linspace(x_min, x_max, n)
    ys = np.linspace(y_min, y_max, n)
    dx = xs[1] - xs[0] if n > 1 else 1.0
    dy = ys[1] - ys[0] if n > 1 else 1.0

    penalties = MANEUVER_PENALTIES.get(boat_type, MANEUVER_PENALTIES["laser"])
    tack_pen = penalties["tack"]
    jibe_pen = penalties["jibe"]

    rows_list = []
    cols_list = []
    costs_list = []

    for di, dj in NEIGHBOR_OFFSETS:
        # Determine valid source indices for this offset
        if di >= 0:
            i_src_start, i_src_end = 0, n - di
        else:
            i_src_start, i_src_end = -di, n
        if dj >= 0:
            j_src_start, j_src_end = 0, n - dj
        else:
            j_src_start, j_src_end = -dj, n

        if i_src_start >= i_src_end or j_src_start >= j_src_end:
            continue

        # Build arrays of source (i, j) indices
        i_src = np.arange(i_src_start, i_src_end)
        j_src = np.arange(j_src_start, j_src_end)
        ii_src, jj_src = np.meshgrid(i_src, j_src, indexing='ij')
        ii_src = ii_src.ravel()
        jj_src = jj_src.ravel()

        ii_dst = ii_src + di
        jj_dst = jj_src + dj

        # Physical coordinates of source and destination
        x_src = xs[ii_src]
        y_src = ys[jj_src]
        x_dst = xs[ii_dst]
        y_dst = ys[jj_dst]

        # Edge distance (constant for a given offset on uniform grid)
        edge_dist = math.sqrt((di * dx) ** 2 + (dj * dy) ** 2)
        if edge_dist < 1e-12:
            continue

        # Heading: compass bearing (0=N, 90=E)
        heading = math.degrees(math.atan2(di * dx, dj * dy)) % 360.0

        # Midpoint coordinates for wind lookup
        mx = 0.5 * (x_src + x_dst)
        my = 0.5 * (y_src + y_dst)

        # Wind at midpoints (vectorized)
        wind_speeds, wind_dirs = wind_field.at_array(mx, my)

        # TWA: absolute angle between heading and wind direction
        twa = np.abs(((heading - wind_dirs) % 360 + 180) % 360 - 180)

        # Clamp TWA to [0, 180]
        twa = np.clip(twa, 0.0, 180.0)

        # Clamp TWS to interpolator range
        tws_clamped = np.clip(wind_speeds, 6.0, 20.0)

        # Boat speed from polars (vectorized)
        pts_query = np.column_stack([tws_clamped, twa])
        boat_speeds = interp(pts_query)

        # Minimum speed floor to avoid infinite costs
        boat_speeds = np.maximum(boat_speeds, 0.1)

        # Base edge time in seconds
        base_time = edge_dist / (boat_speeds * KNOTS_TO_NM_PER_SEC)

        # Determine tack for this edge: starboard(0) if heading clockwise from wind
        cross = ((heading - wind_dirs) % 360 + 360) % 360
        edge_tack = np.where((cross > 0) & (cross < 180), 0, 1).astype(int)

        # Penalty depends on whether this is a tack (upwind) or jibe (downwind)
        maneuver_cost = np.where(twa < 90, tack_pen, jibe_pen)

        # Base spatial node IDs: base_id = i * n + j
        base_src = ii_src * n + jj_src
        base_dst = ii_dst * n + jj_dst

        # Create edges for both source tack states
        for src_tack in [0, 1]:
            penalty = np.where(edge_tack != src_tack, maneuver_cost, 0.0)
            total_cost = base_time + penalty

            src_ids = base_src * 2 + src_tack
            dst_ids = base_dst * 2 + edge_tack

            rows_list.append(src_ids)
            cols_list.append(dst_ids)
            costs_list.append(total_cost)

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_costs = np.concatenate(costs_list)

    total_nodes = n * n * 2
    graph = csr_matrix((all_costs, (all_rows, all_cols)),
                       shape=(total_nodes, total_nodes))

    return xs, ys, graph


def find_nearest_node(xs, ys, x, y):
    """Find grid node (i, j) nearest to point (x, y)."""
    i = int(np.argmin(np.abs(xs - x)))
    j = int(np.argmin(np.abs(ys - y)))
    return i, j


def run_dijkstra(graph, xs, ys, start_xy, end_xy):
    """
    Run Dijkstra on the tack-state-extended graph from start to end.
    Tries both starting tack states and both ending tack states,
    picks the fastest overall.

    Returns
    -------
    path : list of (x, y) tuples along the grid path
    total_time : float, travel time in seconds
    """
    n = len(xs)
    si, sj = find_nearest_node(xs, ys, start_xy[0], start_xy[1])
    ei, ej = find_nearest_node(xs, ys, end_xy[0], end_xy[1])

    base_src = si * n + sj
    base_dst = ei * n + ej

    if base_src == base_dst:
        return [(float(xs[si]), float(ys[sj]))], 0.0

    # Run Dijkstra from both start tack states
    start_nodes = [base_src * 2 + 0, base_src * 2 + 1]
    dist_matrix, predecessors = shortest_path(
        graph, method='D', indices=start_nodes, return_predecessors=True
    )

    # Find best combination of start tack x end tack
    best_cost = float('inf')
    best_src_idx = 0
    best_end_node = base_dst * 2

    for src_idx in [0, 1]:
        for end_tack in [0, 1]:
            end_node = base_dst * 2 + end_tack
            cost = dist_matrix[src_idx, end_node]
            if cost < best_cost:
                best_cost = cost
                best_src_idx = src_idx
                best_end_node = end_node

    total_time = best_cost
    src_node = start_nodes[best_src_idx]
    pred = predecessors[best_src_idx]

    if pred[best_end_node] == -9999:
        # No path found -- fallback to straight line
        return [tuple(start_xy), tuple(end_xy)], float('inf')

    # Reconstruct path from predecessors
    path_ids = []
    current = best_end_node
    while current != src_node and current >= 0 and pred[current] != -9999:
        path_ids.append(current)
        current = int(pred[current])
    if current == src_node:
        path_ids.append(src_node)
    path_ids.reverse()

    # Convert extended node IDs to coordinates (strip tack state)
    path = []
    for nid in path_ids:
        spatial_id = nid // 2
        i = spatial_id // n
        j = spatial_id % n
        path.append((float(xs[i]), float(ys[j])))

    return path, total_time


# ============================================================
# Path to Waypoints Conversion
# ============================================================

def path_to_waypoints(path, wind_field, interp, t_offset, leg_name, target_xy):
    """
    Convert a grid path (list of (x,y)) into waypoint dicts matching the API format.

    Parameters
    ----------
    path : list of (x, y) tuples
    wind_field : WindField
    interp : polar interpolator
    t_offset : float, cumulative time offset at start
    leg_name : str, "upwind" or "downwind"
    target_xy : (x, y) tuple, the destination of this leg (mark or finish)

    Returns
    -------
    waypoints : list of dicts
    """
    if len(path) < 2:
        if len(path) == 1:
            return [{
                "x": path[0][0], "y": path[0][1],
                "heading": 0.0, "tack": "starboard",
                "speed": 0.0, "vmg": 0.0,
                "time": t_offset, "leg": leg_name,
            }]
        return []

    waypoints = []
    t = t_offset

    for idx in range(len(path)):
        px, py = path[idx]

        if idx < len(path) - 1:
            nx, ny = path[idx + 1]
            dx_edge = nx - px
            dy_edge = ny - py
            heading = math.degrees(math.atan2(dx_edge, dy_edge)) % 360.0
        else:
            # Last point: reuse heading from previous edge
            if waypoints:
                heading = waypoints[-1]["heading"]
            else:
                heading = 0.0

        # Local wind at this point
        local_tws, local_wind_dir = wind_field.at(px, py)

        # TWA
        twa = abs(normalize_angle(heading - local_wind_dir))

        # Speed from polars
        speed = get_boat_speed(local_tws, twa, interp)
        speed = max(speed, 0.1)

        # VMG toward target
        dx_to_target = target_xy[0] - px
        dy_to_target = target_xy[1] - py
        dist_to_target = math.sqrt(dx_to_target**2 + dy_to_target**2)
        if dist_to_target > 1e-10:
            bearing_to_target = math.degrees(math.atan2(dx_to_target, dy_to_target)) % 360.0
            angle_diff = math.radians(abs(normalize_angle(heading - bearing_to_target)))
            vmg = speed * math.cos(angle_diff)
        else:
            vmg = 0.0

        # Tack: starboard if heading is clockwise from wind (0 < cross < 180)
        cross = (heading - local_wind_dir) % 360.0
        tack = "starboard" if 0 < cross < 180 else "port"

        # Compute time: for edges after the first point, add travel time
        if idx > 0:
            prev_x, prev_y = path[idx - 1]
            edge_dx = px - prev_x
            edge_dy = py - prev_y
            edge_dist = math.sqrt(edge_dx**2 + edge_dy**2)

            # Use midpoint wind for time computation (matches Dijkstra cost)
            mid_x = 0.5 * (prev_x + px)
            mid_y = 0.5 * (prev_y + py)
            mid_tws, mid_wind_dir = wind_field.at(mid_x, mid_y)
            edge_heading = math.degrees(math.atan2(edge_dx, edge_dy)) % 360.0
            mid_twa = abs(normalize_angle(edge_heading - mid_wind_dir))
            mid_speed = get_boat_speed(mid_tws, mid_twa, interp)
            mid_speed = max(mid_speed, 0.1)
            edge_time = edge_dist / (mid_speed * KNOTS_TO_NM_PER_SEC)
            t += edge_time

        waypoints.append({
            "x": px,
            "y": py,
            "heading": round(heading, 2),
            "tack": tack,
            "speed": round(speed, 3),
            "vmg": round(vmg, 3),
            "time": round(t, 3),
            "leg": leg_name,
        })

    return waypoints


# ============================================================
# User Path with Spatial Wind
# ============================================================

def compute_user_leg_spatial(start, end, via_points, wind_field, interp, t_offset, leg_name, n_samples=40):
    """
    Compute a user-defined multi-segment leg using spatially varying wind.
    Samples ~n_samples points per segment for smooth time integration.

    Parameters
    ----------
    start : np.array, start point
    end : np.array, end point
    via_points : list of np.array, intermediate waypoints
    wind_field : WindField
    interp : polar interpolator
    t_offset : float, cumulative time offset at start
    leg_name : str, "upwind" or "downwind"
    n_samples : int, samples per segment

    Returns
    -------
    dict with keys: waypoints, distance, time, n_maneuvers
    """
    all_wps = []
    total_dist = 0.0
    t = t_offset

    nodes = [start] + via_points + [end]
    segments = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    for seg_start, seg_end in segments:
        diff = seg_end - seg_start
        dist = float(np.linalg.norm(diff))
        if dist < 1e-10:
            continue

        heading = math.degrees(math.atan2(diff[0], diff[1])) % 360.0

        for s in range(n_samples + 1):
            frac = s / n_samples
            px = seg_start[0] + frac * diff[0]
            py = seg_start[1] + frac * diff[1]

            # Local wind
            local_tws, local_wind_dir = wind_field.at(px, py)

            # TWA
            twa = abs(normalize_angle(heading - local_wind_dir))

            # Speed
            speed = get_boat_speed(local_tws, twa, interp)
            speed = max(speed, 0.1)

            # Tack
            cross = (heading - local_wind_dir) % 360.0
            tack = "starboard" if 0 < cross < 180 else "port"

            # VMG (along heading direction)
            vmg = speed * math.cos(math.radians(twa))

            # Time integration using midpoint wind
            if s > 0:
                step_dist = dist / n_samples
                mid_frac = (s - 0.5) / n_samples
                mid_x = seg_start[0] + mid_frac * diff[0]
                mid_y = seg_start[1] + mid_frac * diff[1]
                mid_tws, mid_wind_dir = wind_field.at(mid_x, mid_y)
                mid_twa = abs(normalize_angle(heading - mid_wind_dir))
                mid_speed = get_boat_speed(mid_tws, mid_twa, interp)
                mid_speed = max(mid_speed, 0.1)
                step_time = step_dist / (mid_speed * KNOTS_TO_NM_PER_SEC)
                t += step_time

            all_wps.append({
                "x": float(px),
                "y": float(py),
                "heading": round(heading, 2),
                "tack": tack,
                "speed": round(speed, 3),
                "vmg": round(abs(vmg), 3),
                "time": round(t, 3),
                "leg": leg_name,
            })

        total_dist += dist

    total_time = (all_wps[-1]["time"] - t_offset) if all_wps else 0.0

    # Count actual tack/jibe changes
    n_maneuvers = 0
    for i in range(1, len(all_wps)):
        if all_wps[i]["tack"] != all_wps[i - 1]["tack"]:
            n_maneuvers += 1

    return {"waypoints": all_wps, "distance": total_dist, "time": total_time, "n_maneuvers": n_maneuvers}


# ============================================================
# Main Course Computation
# ============================================================

def compute_full_course(body):
    boat_type = body.get("boat_type", "laser")
    base_tws = body.get("wind_speed", 12.0)
    base_wind_dir = body.get("wind_direction", 0.0)

    start = np.array([body.get("start_x", 0.0), body.get("start_y", 0.0)])
    mark = np.array([body.get("mark_x", 0.0), body.get("mark_y", 1.0)])
    finish = np.array([body.get("finish_x", 0.0), body.get("finish_y", 0.0)])

    interp = get_interpolator(boat_type)

    # Create spatially-varying wind field
    seed = make_wind_seed(body)
    wind_field = WindField(base_tws, base_wind_dir, seed=seed)

    # Build grid and graph covering all course points
    course_points = [
        (float(start[0]), float(start[1])),
        (float(mark[0]), float(mark[1])),
        (float(finish[0]), float(finish[1])),
    ]
    xs, ys, graph = build_grid_and_graph(course_points, wind_field, interp, boat_type)

    # Run Dijkstra: upwind (start -> mark)
    upwind_path, upwind_time = run_dijkstra(
        graph, xs, ys,
        (float(start[0]), float(start[1])),
        (float(mark[0]), float(mark[1])),
    )

    # Run Dijkstra: downwind (mark -> finish) on the SAME graph
    downwind_path, downwind_time = run_dijkstra(
        graph, xs, ys,
        (float(mark[0]), float(mark[1])),
        (float(finish[0]), float(finish[1])),
    )

    # Convert paths to waypoints
    upwind_wps = path_to_waypoints(
        upwind_path, wind_field, interp,
        t_offset=0.0, leg_name="upwind",
        target_xy=(float(mark[0]), float(mark[1])),
    )

    t_after_up = upwind_wps[-1]["time"] if upwind_wps else 0.0

    downwind_wps = path_to_waypoints(
        downwind_path, wind_field, interp,
        t_offset=t_after_up, leg_name="downwind",
        target_xy=(float(finish[0]), float(finish[1])),
    )

    all_wps = upwind_wps + downwind_wps

    # Distances
    def path_distance(path):
        d = 0.0
        for i in range(1, len(path)):
            ddx = path[i][0] - path[i - 1][0]
            ddy = path[i][1] - path[i - 1][1]
            d += math.sqrt(ddx**2 + ddy**2)
        return d

    upwind_dist = path_distance(upwind_path)
    downwind_dist = path_distance(downwind_path)
    total_dist = upwind_dist + downwind_dist
    total_time = all_wps[-1]["time"] if all_wps else 0.0

    # Count tack/jibe changes
    def count_maneuvers(waypoints):
        changes = 0
        for i in range(1, len(waypoints)):
            if waypoints[i]["tack"] != waypoints[i - 1]["tack"]:
                changes += 1
        return changes

    n_tacks = count_maneuvers(upwind_wps)
    n_jibes = count_maneuvers(downwind_wps)

    # Compute laylines from wind at mark position (backward compat)
    mark_tws, mark_wind_dir = wind_field.at(float(mark[0]), float(mark[1]))
    up_twa, up_speed, up_vmg = find_optimal_vmg(mark_tws, interp)
    dn_twa, dn_speed, dn_vmg = find_optimal_downwind_vmg(mark_tws, interp)

    laylines = {
        "upwind_sb": (mark_wind_dir + up_twa) % 360,
        "upwind_port": (mark_wind_dir - up_twa) % 360,
        "downwind_sb": (mark_wind_dir + dn_twa) % 360,
        "downwind_port": (mark_wind_dir - dn_twa) % 360,
    }

    # Wind visualization grid
    x_min_grid = float(xs[0])
    x_max_grid = float(xs[-1])
    y_min_grid = float(ys[0])
    y_max_grid = float(ys[-1])
    wind_grid = wind_field.get_grid(x_min_grid, x_max_grid, y_min_grid, y_max_grid, n=12)

    result = {
        "waypoints": all_wps,
        "summary": {
            "total_distance_nm": round(total_dist, 6),
            "total_time_s": round(total_time, 3),
            "upwind_twa": up_twa,
            "upwind_vmg": round(up_vmg, 3),
            "n_tacks": n_tacks,
            "downwind_twa": dn_twa,
            "downwind_vmg": round(dn_vmg, 3),
            "n_jibes": n_jibes,
            "laylines": laylines,
        },
        "wind_grid": wind_grid,
    }

    # User challenge path (multi-waypoint format)
    if "user_upwind_points" in body or "user_downwind_points" in body:
        upwind_pts = body.get("user_upwind_points", [])
        downwind_pts = body.get("user_downwind_points", [])

        upwind_via = [np.array(p) for p in upwind_pts]
        downwind_via = [np.array(p) for p in downwind_pts]

        u_up = compute_user_leg_spatial(
            start, mark, upwind_via, wind_field, interp,
            t_offset=0.0, leg_name="upwind",
        )
        u_t_after = u_up["waypoints"][-1]["time"] if u_up["waypoints"] else 0.0
        u_dn = compute_user_leg_spatial(
            mark, finish, downwind_via, wind_field, interp,
            t_offset=u_t_after, leg_name="downwind",
        )

        u_all_wps = u_up["waypoints"] + u_dn["waypoints"]
        result["user_waypoints"] = u_all_wps
        result["user_summary"] = {
            "total_distance_nm": round(u_up["distance"] + u_dn["distance"], 6),
            "total_time_s": round(u_all_wps[-1]["time"], 3) if u_all_wps else 0.0,
            "n_tacks": u_up["n_maneuvers"],
            "n_jibes": u_dn["n_maneuvers"],
        }

    # Legacy single-point format (backward compat)
    elif "user_tack_x" in body and "user_jibe_x" in body:
        user_tack = np.array([body["user_tack_x"], body["user_tack_y"]])
        user_jibe = np.array([body["user_jibe_x"], body["user_jibe_y"]])

        u_up = compute_user_leg_spatial(
            start, mark, [user_tack], wind_field, interp,
            t_offset=0.0, leg_name="upwind",
        )
        u_t_after = u_up["waypoints"][-1]["time"] if u_up["waypoints"] else 0.0
        u_dn = compute_user_leg_spatial(
            mark, finish, [user_jibe], wind_field, interp,
            t_offset=u_t_after, leg_name="downwind",
        )

        u_all_wps = u_up["waypoints"] + u_dn["waypoints"]
        result["user_waypoints"] = u_all_wps
        result["user_summary"] = {
            "total_distance_nm": round(u_up["distance"] + u_dn["distance"], 6),
            "total_time_s": round(u_all_wps[-1]["time"], 3) if u_all_wps else 0.0,
            "n_tacks": u_up["n_maneuvers"],
            "n_jibes": u_dn["n_maneuvers"],
        }

    return result


# ============================================================
# Vercel Serverless Handler
# ============================================================

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            result = compute_full_course(body)
            self._respond(200, result)
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _respond(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
