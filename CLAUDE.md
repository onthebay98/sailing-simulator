# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
python3 sailing_sim.py              # opens browser at http://localhost:8420
python3 sailing_sim.py --port 9000  # custom port
```

No virtual environment needed. Dependencies: numpy, scipy (system-installed). No matplotlib — the UI is a self-contained HTML5 Canvas app served by Python's stdlib `http.server`.

## Architecture

Single-file app (`sailing_sim.py`) organized into 7 sections:

1. **Constants** — unit conversions
2. **Dataclasses** — `SimConfig` (frozen, includes `boat_type`), `WindState`, `Waypoint`, `SailingPath`
3. **Polar Speed Model** — `LASER_POLAR_DATA`, `FOUR_TWENTY_POLAR_DATA`, and `J24_POLAR_DATA` dicts, `BOAT_POLARS` registry, cached `RegularGridInterpolator` per boat type. TWS range 6-20 kts. No-go zone enforced by data.
4. **Wind Model** — `WindField` class with spatially-varying wind (sinusoidal perturbations for smooth variation). `make_wind_seed()` generates deterministic seeds from course parameters.
5. **VMG Optimizer & Path Planner** — Grid-based Dijkstra (weather routing) with tack-state-extended graph. `MANEUVER_PENALTIES` dict per boat type. `build_course_graph()` creates a sparse graph with `(i, j, tack)` state nodes. `find_optimal_grid_path()` runs shortest-path. `compute_user_leg_spatial()` and `compute_user_course_spatial()` compute paths through user-specified waypoint lists.
6. **Embedded Web App** — `APP_HTML` string constant containing the full HTML/JS/CSS frontend. Canvas-based animation with two modes: Simulate (optimal path) and Challenge (user vs optimal).
7. **HTTP Server** — `SimHandler` serves `GET /` (HTML) and `POST /compute` (JSON path computation). Auto-opens browser on startup.

Vercel deployment uses separate files: `api/compute.py` (serverless function) and `public/index.html` (frontend). These must be kept in sync with `sailing_sim.py`.

Previous version preserved in `v1/` directory and git tag `v1-constant-wind`.

## Modes

- **Simulate**: Shows the optimal path with animation.
- **Challenge (Tactics Test)**: User clicks canvas to place multiple tack/jibe waypoints, then races against the optimal ghost boat. Score = `optimal_time / user_time * 100%`. Both boats animate simultaneously with time-synchronized playback.

## Boat Types

- **Laser**: Upwind VMG angle ~44.5°, downwind ~144.5°. Moderate upwind polar. Tack penalty: 6s, jibe: 5s.
- **420**: Upwind VMG angle ~40°, downwind ~146°. Flatter upwind polar. Faster on reaches (trapeze/spinnaker). Tack penalty: 8s, jibe: 10s.
- **J/24**: Upwind VMG angle ~48°, downwind ~140°. Steeper upwind polar, heavier displacement keelboat. Tack penalty: 12s, jibe: 10s.

## Wind Model

Spatially-varying wind field using sinusoidal perturbations. Both wind speed and direction vary smoothly across the course area. A deterministic seed is generated from course parameters via MD5 hash. Wind arrows on canvas show the local wind field.

## Coordinate Conventions

- Positions: (x, y) in nautical miles. x = East, y = North.
- Angles: compass bearings (0° = North, 90° = East, clockwise). Vectors: `dx = sin(bearing)`, `dy = cos(bearing)`.
- Wind direction: meteorological "FROM" convention. `wind_direction_deg=0` = wind from north.
- Canvas y-axis is flipped: world y increases north, canvas y increases down. Downwind arrow direction in canvas: `(-sin(rad), +cos(rad))`.

## API

`POST /compute` with JSON body:
```json
{"boat_type": "laser", "wind_speed": 12, "wind_direction": 0, "start_x": 0, "start_y": 0, "mark_x": 0, "mark_y": 1, "finish_x": 0.15, "finish_y": 0}
```
Returns `{"waypoints": [...], "summary": {...}, "wind_grid": {...}}`. Course: start → upwind mark → finish (downwind).

Optional challenge mode fields (multi-waypoint):
```json
{"user_upwind_points": [[0.2, 0.4], [-0.1, 0.7]], "user_downwind_points": [[0.1, 0.5]]}
```
When present, response includes `"user_waypoints"` and `"user_summary"` alongside the optimal path.

Summary always includes `"laylines"` with upwind/downwind starboard/port headings.

## Frontend Controls

- **Mode**: Simulate or Challenge toggle
- **Boat**: Laser, 420, J/24
- **Wind Speed**: 6-20 kts slider
- **Wind Direction**: -90° to +90° slider
- **Start/Mark/Finish**: position sliders (hidden in Challenge mode)
- **Laylines**: checkbox to show optimal tack/jibe angle lines from mark
- **New Course**: randomize course parameters (Challenge mode)

### Challenge Mode Controls

- **Click canvas**: Place waypoint (tack or jibe depending on phase)
- **Upwind/Downwind button**: Toggle between upwind (tack) and downwind (jibe) placement phase
- **Undo button** (or Backspace): Remove last placed waypoint in current phase

## Planned Extensions

- Time-varying wind direction (shifts)
- Puffs (localized TWS increases, position-dependent)
- Start line (line segment instead of single point)
- Multiple boats / race simulation
- Score history tracking across challenge attempts
