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
# Section 4: Wind Model
# ============================================================

def make_wind_fn(config: SimConfig) -> Callable[[float, float, float], WindState]:
    """Return a wind function. Currently constant; future: time/position varying."""
    constant_wind = WindState(
        speed_kts=config.wind_speed_kts,
        direction_deg=config.wind_direction_deg,
    )

    def get_wind(time: float, x: float, y: float) -> WindState:
        return constant_wind

    return get_wind


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
    """Compute optimal path for a full upwind-downwind course."""
    start = np.array([config.start_x, config.start_y])
    mark = np.array([config.mark_x, config.mark_y])
    finish = np.array([config.finish_x, config.finish_y])

    wind_from = config.wind_direction_deg
    tws = config.wind_speed_kts

    upwind = compute_leg_path(
        start, mark, wind_from, tws, config.boat_type,
        config.dt_seconds, t_offset=0.0, leg_name="upwind",
    )

    t_after_upwind = upwind.waypoints[-1].elapsed_seconds if upwind.waypoints else 0.0

    downwind = compute_leg_path(
        mark, finish, wind_from, tws, config.boat_type,
        config.dt_seconds, t_offset=t_after_upwind, leg_name="downwind",
    )

    all_waypoints = upwind.waypoints + downwind.waypoints

    return {
        "waypoints": all_waypoints,
        "upwind": upwind,
        "downwind": downwind,
        "total_distance_nm": upwind.total_distance_nm + downwind.total_distance_nm,
        "total_time_seconds": (upwind.total_time_seconds + downwind.total_time_seconds),
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
    n_steps = max(1, int(math.ceil(leg_time / dt)))
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
.ctrl select,.ctrl input[type=range]{background:#0d1f3c;color:#ccddee;border:1px solid #1a3355;
  border-radius:4px;padding:4px 8px;font-family:inherit;font-size:13px}
.ctrl select{cursor:pointer;min-width:80px}
.ctrl input[type=range]{width:140px;cursor:pointer;accent-color:#44aaff}
.ctrl .val{font-size:12px;color:#88bbdd;min-width:40px;text-align:right}
.ctrl-row{display:flex;align-items:center;gap:6px}
#simulate-btn{background:#1a5599;color:#eef;border:none;border-radius:6px;padding:8px 24px;
  font-family:inherit;font-size:13px;cursor:pointer;font-weight:600;letter-spacing:0.5px;
  transition:background 0.15s}
#simulate-btn:hover{background:#2266bb}
#simulate-btn:active{background:#0e3d77}
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
#summary{position:absolute;top:120px;right:12px;background:rgba(10,22,40,0.85);
  padding:10px 14px;border-radius:6px;border:1px solid #1a2a44;font-size:12px;
  line-height:1.6;pointer-events:none;min-width:200px}
#summary .lbl{color:#6688aa}
#summary .val{color:#ccddee}
#finish-overlay{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-size:28px;font-weight:700;color:#00ff88;pointer-events:none;opacity:0;
  text-shadow:0 0 20px rgba(0,255,136,0.4);transition:opacity 0.5s}
</style>
</head>
<body>
<div id="controls">
  <div class="ctrl">
    <label>Boat</label>
    <select id="boat"><option value="laser">Laser</option><option value="420">420</option><option value="j24">J/24</option></select>
  </div>
  <div class="ctrl">
    <label>Wind Direction</label>
    <div class="ctrl-row">
      <input type="range" id="windDir" min="-180" max="179" step="1" value="0">
      <span class="val" id="windDirVal">000&deg;</span>
    </div>
  </div>
  <div class="ctrl">
    <label>Start</label>
    <div class="ctrl-row">
      <input type="range" id="startX" min="-0.5" max="0.5" step="0.05" value="0">
      <span class="val" id="startXVal">0.00</span>
    </div>
  </div>
  <div class="ctrl">
    <label>Mark</label>
    <div class="ctrl-row">
      <input type="range" id="markX" min="-0.5" max="0.5" step="0.05" value="0">
      <span class="val" id="markXVal">0.00</span>
    </div>
  </div>
  <div class="ctrl">
    <label>Finish</label>
    <div class="ctrl-row">
      <input type="range" id="finishX" min="-0.5" max="0.5" step="0.05" value="0">
      <span class="val" id="finishXVal">0.00</span>
    </div>
  </div>
  <button id="simulate-btn">Simulate</button>
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
</div>
<script>
const $ = id => document.getElementById(id);

const canvas = $('sim');
const ctx = canvas.getContext('2d');
let waypoints = null, summary = null, frame = 0, animId = null;

// --- State: 'idle' | 'running' | 'paused' | 'finished' ---
let simState = 'idle';
const controls = ['boat','windDir','startX','markX','finishX'];

function setControlsEnabled(enabled) {
  for (const id of controls) {
    $(id).disabled = !enabled;
  }
  $('controls').style.opacity = enabled ? '1' : '0.5';
}

// --- Slider live updates + course redraw ---
for (const [id, valId, fmt] of [
  ['windDir','windDirVal', v => String(((+v)%360+360)%360).padStart(3,'0')+'°'],
  ['startX','startXVal', v => parseFloat(v).toFixed(2)],
  ['markX','markXVal', v => parseFloat(v).toFixed(2)],
  ['finishX','finishXVal', v => parseFloat(v).toFixed(2)],
]) {
  $(id).addEventListener('input', () => {
    $(valId).textContent = fmt($(id).value);
    if (simState === 'idle') drawCourse();
  });
}
$('boat').addEventListener('change', () => {
  if (simState === 'idle') drawCourse();
});

// --- Read current slider values ---
function getParams() {
  return {
    startX: parseFloat($('startX').value),
    markX: parseFloat($('markX').value),
    markY: 1.0,
    finishX: parseFloat($('finishX').value),
    finishY: 0.0,
    windDir: ((parseFloat($('windDir').value) % 360) + 360) % 360,
    windSpeed: 12,
  };
}

// --- Canvas setup ---
function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  canvas.width = r.width * devicePixelRatio;
  canvas.height = r.height * devicePixelRatio;
  canvas.style.width = r.width + 'px';
  canvas.style.height = r.height + 'px';
  if (waypoints) drawScene(Math.min(frame, waypoints.length - 1));
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
  // Center the view: make x range symmetric around the midpoint
  const xMid = (xMin + xMax) / 2, yMid = (yMin + yMax) / 2;
  const halfX = (xMax - xMin) / 2, halfY = (yMax - yMin) / 2;
  const half = Math.max(halfX, halfY, 0.15);
  const pad = 1.3; // 30% padding
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

// --- Draw wind arrow (used by both course preview and animation) ---
function drawWindArrow(dir, speed) {
  const dpr = devicePixelRatio;
  const cw = canvas.width;
  const windRad = dir * Math.PI / 180;
  const arrowLen = 40 * dpr;
  const cx = cw - 55*dpr, cy = 55*dpr;

  // Circle background
  ctx.strokeStyle = 'rgba(136,187,255,0.3)';
  ctx.lineWidth = 1.5*dpr;
  ctx.beginPath(); ctx.arc(cx, cy, arrowLen*0.65, 0, Math.PI*2); ctx.stroke();

  // Arrow body: points in direction wind blows TO (opposite of FROM)
  const blowDx = -Math.sin(windRad), blowDy = -Math.cos(windRad);
  // On canvas: +x is right, +y is down, so flip dy
  const ax1 = cx - blowDx*arrowLen*0.45, ay1 = cy + blowDy*arrowLen*0.45;
  const ax2 = cx + blowDx*arrowLen*0.45, ay2 = cy - blowDy*arrowLen*0.45;
  ctx.strokeStyle = '#88bbff';
  ctx.lineWidth = 2.5*dpr;
  ctx.beginPath(); ctx.moveTo(ax1,ay1); ctx.lineTo(ax2,ay2); ctx.stroke();

  // Arrowhead
  const angle = Math.atan2(ay2-ay1, ax2-ax1);
  const hs = 10*dpr;
  ctx.beginPath();
  ctx.moveTo(ax2, ay2);
  ctx.lineTo(ax2 - hs*Math.cos(angle-0.4), ay2 - hs*Math.sin(angle-0.4));
  ctx.moveTo(ax2, ay2);
  ctx.lineTo(ax2 - hs*Math.cos(angle+0.4), ay2 - hs*Math.sin(angle+0.4));
  ctx.stroke();

  // Label
  ctx.fillStyle = '#88bbff';
  ctx.font = (11*dpr)+'px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(speed+' kts', cx, cy + arrowLen*0.95);
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
  // Start (green circle)
  const [sx,sy] = w2c(p.startX, 0, b);
  ctx.fillStyle = '#00dd66';
  ctx.beginPath(); ctx.arc(sx, sy, 7*dpr, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = 'rgba(0,221,102,0.6)';
  ctx.font = (10*dpr)+'px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('START', sx, sy + 14*dpr);

  // Upwind mark (orange diamond)
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

  // Finish (checkered flag style - red square)
  const [fx,fy] = w2c(p.finishX, p.finishY, b);
  ctx.fillStyle = '#ff4466';
  ctx.beginPath(); ctx.arc(fx, fy, 7*dpr, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = 'rgba(255,68,102,0.6)';
  ctx.font = (10*dpr)+'px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('FINISH', fx, fy + 14*dpr);
}

// --- Course preview (no simulation, just marks + wind) ---
function drawCourse() {
  if (animId) return;
  const p = getParams();
  const cw = canvas.width, ch = canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#0d1f3c';
  ctx.fillRect(0, 0, cw, ch);

  const b = makeBounds([[p.startX, 0], [p.markX, p.markY], [p.finishX, p.finishY]]);
  drawGrid(b);
  drawMarks(p, b);
  drawWindArrow(p.windDir, p.windSpeed);
}

// --- Full animation scene ---
function drawScene(idx) {
  const dpr = devicePixelRatio;
  const cw = canvas.width, ch = canvas.height;
  const p = getParams();
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#0d1f3c';
  ctx.fillRect(0, 0, cw, ch);

  // Bounds from waypoints
  const pts = waypoints.map(w => [w.x, w.y]);
  pts.push([p.startX, 0], [p.markX, p.markY], [p.finishX, p.finishY]);
  const b = makeBounds(pts);
  const wp = waypoints[idx];

  drawGrid(b);

  // Planned path (dashed)
  ctx.setLineDash([6*dpr, 4*dpr]);
  ctx.strokeStyle = 'rgba(68,85,102,0.5)';
  ctx.lineWidth = 1.5*dpr;
  ctx.beginPath();
  for (let i = 0; i < waypoints.length; i++) {
    const [px,py] = w2c(waypoints[i].x, waypoints[i].y, b);
    i === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Tack/jibe points and mark rounding
  for (let i = 1; i < waypoints.length; i++) {
    // Leg transition (mark rounding)
    if (waypoints[i].leg !== waypoints[i-1].leg) {
      const [tx,ty] = w2c(waypoints[i].x, waypoints[i].y, b);
      ctx.fillStyle = '#ff8833';
      ctx.font = (10*dpr)+'px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('ROUNDING', tx, ty - 14*dpr);
    }
    // Tack or jibe within a leg
    else if (waypoints[i].tack !== waypoints[i-1].tack) {
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
  drawWindArrow(p.windDir, p.windSpeed);

  // Wake trail
  const trailStart = Math.max(0, idx - 300);
  if (idx > 0) {
    ctx.strokeStyle = 'rgba(68,221,255,0.4)';
    ctx.lineWidth = 2*dpr;
    ctx.beginPath();
    for (let i = trailStart; i <= idx; i++) {
      const [px,py] = w2c(waypoints[i].x, waypoints[i].y, b);
      i === trailStart ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
    }
    ctx.stroke();
  }

  // Boat triangle
  const [bx,by] = w2c(wp.x, wp.y, b);
  const headingRad = (90 - wp.heading) * Math.PI / 180;
  const bs = 10*dpr;
  ctx.save();
  ctx.translate(bx, by);
  ctx.rotate(-headingRad);
  ctx.fillStyle = 'white';
  ctx.beginPath();
  ctx.moveTo(0, -bs);
  ctx.lineTo(-bs*0.6, bs*0.5);
  ctx.lineTo(bs*0.6, bs*0.5);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// --- Info panel ---
function updateInfo(idx) {
  const wp = waypoints[idx];
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

// --- Button: Simulate / Pause / Restart ---
let stepsPerFrame = 1;
const btn = $('simulate-btn');
btn.addEventListener('click', handleBtn);

function stopAnim() {
  if (animId) { cancelAnimationFrame(animId); animId = null; }
}

function goIdle() {
  stopAnim();
  simState = 'idle';
  waypoints = null;
  summary = null;
  btn.textContent = 'Simulate';
  setControlsEnabled(true);
  $('finish-overlay').style.opacity = '0';
  $('summary').style.display = 'none';
  drawCourse();
}

function handleBtn() {
  if (simState === 'idle') {
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
    goIdle();
  }
}

async function fetchAndRun() {
  btn.textContent = '...';
  btn.disabled = true;
  $('finish-overlay').style.opacity = '0';
  setControlsEnabled(false);
  const p = getParams();
  const body = {
    boat_type: $('boat').value,
    wind_speed: p.windSpeed,
    wind_direction: p.windDir,
    start_x: p.startX,
    mark_x: p.markX,
    mark_y: p.markY,
    finish_x: p.finishX,
    finish_y: p.finishY,
  };
  try {
    const resp = await fetch('/compute', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    waypoints = data.waypoints;
    summary = data.summary;
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
    startAnimation();
  } catch(e) {
    console.error(e);
    btn.disabled = false;
    goIdle();
  }
}

function startAnimation() {
  stopAnim();
  frame = 0;
  simState = 'running';
  const targetWallMs = 20000;
  stepsPerFrame = Math.max(1, Math.floor(waypoints.length / (targetWallMs / 16.67)));
  btn.textContent = 'Pause';
  tickLoop();
}

function tickLoop() {
  function tick() {
    const idx = Math.min(frame, waypoints.length - 1);
    drawScene(idx);
    updateInfo(idx);
    if (idx < waypoints.length - 1) {
      frame += stepsPerFrame;
      animId = requestAnimationFrame(tick);
    } else {
      animId = null;
      simState = 'finished';
      btn.textContent = 'Reset';
      $('finish-overlay').textContent = 'FINISHED  ' + fmtTime(summary.total_time_s);
      $('finish-overlay').style.opacity = '1';
    }
  }
  animId = requestAnimationFrame(tick);
}

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
                mark_x=body.get("mark_x", 0.0),
                mark_y=body.get("mark_y", 1.0),
                finish_x=body.get("finish_x", 0.0),
                finish_y=body.get("finish_y", 0.0),
            )
            course = compute_full_course(config)
            up = course["upwind"]
            down = course["downwind"]

            result = {
                "waypoints": [
                    {"x": wp.x, "y": wp.y, "heading": wp.heading_deg,
                     "tack": wp.tack, "speed": wp.speed_kts,
                     "vmg": wp.vmg_kts, "time": wp.elapsed_seconds,
                     "leg": wp.leg}
                    for wp in course["waypoints"]
                ],
                "summary": {
                    "total_distance_nm": course["total_distance_nm"],
                    "total_time_s": course["total_time_seconds"],
                    "upwind_twa": up.optimal_twa_deg,
                    "upwind_vmg": up.optimal_vmg_kts,
                    "upwind_speed": up.optimal_speed_kts,
                    "n_tacks": up.n_tacks,
                    "downwind_twa": down.optimal_twa_deg,
                    "downwind_vmg": down.optimal_vmg_kts,
                    "downwind_speed": down.optimal_speed_kts,
                    "n_jibes": down.n_tacks,
                },
            }
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
