import json
import math
import sys
import os
from http.server import BaseHTTPRequestHandler

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# Constants
# ============================================================

KNOTS_TO_NM_PER_SEC = 1.0 / 3600.0

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
# VMG Optimizer & Path Planner
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


def compute_tack_point(start, target, heading1_deg, heading2_deg):
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


def discretize_leg(start, end, heading_deg, tack, speed_kts, vmg_kts, dt, t_offset, leg):
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
        waypoints.append({
            "x": float(pos[0]), "y": float(pos[1]),
            "heading": heading_deg, "tack": tack,
            "speed": speed_kts, "vmg": vmg_kts,
            "time": t_offset + frac * leg_time, "leg": leg,
        })
    return waypoints


def compute_leg_path(start, target, wind_from, tws, boat_type, dt, t_offset=0.0, leg_name="upwind"):
    interp = get_interpolator(boat_type)
    diff = target - start
    target_bearing = math.degrees(math.atan2(diff[0], diff[1])) % 360
    total_straight_dist = float(np.linalg.norm(diff))

    if total_straight_dist < 1e-10:
        return {"waypoints": [], "optimal_twa": 0, "optimal_vmg": 0, "optimal_speed": 0, "n_maneuvers": 0, "distance": 0}

    angle_off_wind = abs(normalize_angle(target_bearing - wind_from))

    if angle_off_wind <= 90:
        opt_twa, opt_speed, opt_vmg = find_optimal_vmg(tws, interp)
    else:
        opt_twa, opt_speed, opt_vmg = find_optimal_downwind_vmg(tws, interp)

    starboard_heading = (wind_from + opt_twa) % 360
    port_heading = (wind_from - opt_twa) % 360

    direct_speed = get_boat_speed(tws, angle_off_wind, interp)
    direct_time = total_straight_dist / (direct_speed * KNOTS_TO_NM_PER_SEC) if direct_speed > 1e-6 else float("inf")

    tp_a, dist_a1, dist_a2 = compute_tack_point(start, target, starboard_heading, port_heading)
    tp_b, dist_b1, dist_b2 = compute_tack_point(start, target, port_heading, starboard_heading)

    path_a_valid = dist_a1 > 1e-6 and dist_a2 > 1e-6
    path_b_valid = dist_b1 > 1e-6 and dist_b2 > 1e-6

    time_a = (dist_a1 + dist_a2) / (opt_speed * KNOTS_TO_NM_PER_SEC) if path_a_valid else float("inf")
    time_b = (dist_b1 + dist_b2) / (opt_speed * KNOTS_TO_NM_PER_SEC) if path_b_valid else float("inf")
    best_two_time = min(time_a, time_b)

    if direct_time <= best_two_time:
        cross = normalize_angle(target_bearing - wind_from)
        tack = "starboard" if cross > 0 else "port"
        actual_vmg = direct_speed * math.cos(math.radians(angle_off_wind))
        waypoints = discretize_leg(start, target, target_bearing, tack, direct_speed, abs(actual_vmg), dt, t_offset, leg_name)
        return {
            "waypoints": waypoints,
            "optimal_twa": opt_twa, "optimal_vmg": opt_vmg, "optimal_speed": opt_speed,
            "n_maneuvers": 0, "distance": total_straight_dist,
        }

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

    wp1 = discretize_leg(start, tack_point, leg1_heading, leg1_tack, opt_speed, opt_vmg, dt, t_offset, leg_name)
    t_after = wp1[-1]["time"] if wp1 else t_offset
    wp2 = discretize_leg(tack_point, target, leg2_heading, leg2_tack, opt_speed, opt_vmg, dt, t_after, leg_name)

    return {
        "waypoints": wp1 + wp2,
        "optimal_twa": opt_twa, "optimal_vmg": opt_vmg, "optimal_speed": opt_speed,
        "n_maneuvers": 1, "distance": leg1_dist + leg2_dist,
    }


def compute_full_course(body):
    boat_type = body.get("boat_type", "laser")
    tws = body.get("wind_speed", 12.0)
    wind_from = body.get("wind_direction", 0.0)
    dt = 1.0

    start = np.array([body.get("start_x", 0.0), body.get("start_y", 0.0)])
    mark = np.array([body.get("mark_x", 0.0), body.get("mark_y", 1.0)])
    finish = np.array([body.get("finish_x", 0.0), body.get("finish_y", 0.0)])

    up = compute_leg_path(start, mark, wind_from, tws, boat_type, dt, t_offset=0.0, leg_name="upwind")
    t_after_up = up["waypoints"][-1]["time"] if up["waypoints"] else 0.0
    down = compute_leg_path(mark, finish, wind_from, tws, boat_type, dt, t_offset=t_after_up, leg_name="downwind")

    all_wps = up["waypoints"] + down["waypoints"]
    total_dist = up["distance"] + down["distance"]
    total_time = all_wps[-1]["time"] if all_wps else 0.0

    return {
        "waypoints": all_wps,
        "summary": {
            "total_distance_nm": total_dist,
            "total_time_s": total_time,
            "upwind_twa": up["optimal_twa"],
            "upwind_vmg": up["optimal_vmg"],
            "upwind_speed": up["optimal_speed"],
            "n_tacks": up["n_maneuvers"],
            "downwind_twa": down["optimal_twa"],
            "downwind_vmg": down["optimal_vmg"],
            "downwind_speed": down["optimal_speed"],
            "n_jibes": down["n_maneuvers"],
        },
    }


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
