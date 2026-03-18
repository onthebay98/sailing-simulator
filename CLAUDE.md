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
3. **Polar Speed Model** — `LASER_POLAR_DATA` and `FOUR_TWENTY_POLAR_DATA` dicts, `BOAT_POLARS` registry, cached `RegularGridInterpolator` per boat type. TWS range 6-20 kts. No-go zone enforced by data.
4. **Wind Model** — `make_wind_fn()` returns `(time, x, y) -> WindState`. Currently constant; hook for future variability.
5. **VMG Optimizer & Path Planner** — `find_optimal_vmg()` sweeps upwind TWA, `find_optimal_downwind_vmg()` sweeps downwind TWA. `compute_leg_path()` handles both upwind (tacking) and downwind (jibing) via 2×2 ray intersection. `compute_full_course()` chains upwind + downwind legs.
6. **Embedded Web App** — `APP_HTML` string constant containing the full HTML/JS/CSS frontend. Canvas-based animation with controls (boat type, wind direction, start/mark/finish positions).
7. **HTTP Server** — `SimHandler` serves `GET /` (HTML) and `POST /compute` (JSON path computation). Auto-opens browser on startup.

## Boat Types

- **Laser**: Upwind VMG angle ~44.5°, downwind ~144.5°. Moderate upwind polar.
- **420**: Upwind VMG angle ~40°, downwind ~146°. Flatter upwind polar (less speed gain from bearing off). Faster on reaches due to trapeze/spinnaker.
- **J/24**: Upwind VMG angle ~48°, downwind ~140°. Steeper upwind polar, heavier displacement keelboat.

## Coordinate Conventions

- Positions: (x, y) in nautical miles. x = East, y = North.
- Angles: compass bearings (0° = North, 90° = East, clockwise). Vectors: `dx = sin(bearing)`, `dy = cos(bearing)`.
- Wind direction: meteorological "FROM" convention. `wind_direction_deg=0` = wind from north.

## API

`POST /compute` with JSON body:
```json
{"boat_type": "laser", "wind_speed": 12, "wind_direction": 0, "start_x": 0, "mark_x": 0, "mark_y": 1, "finish_x": 0, "finish_y": 0}
```
Returns `{"waypoints": [...], "summary": {...}}`. Course: start → upwind mark → finish (downwind).

## Planned Extensions

- Time-varying wind direction (shifts) via `make_wind_fn()`
- Puffs (localized TWS increases, position-dependent)
- Start line (line segment instead of single point)
- Multiple boats / race simulation
