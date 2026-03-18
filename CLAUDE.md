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
5. **VMG Optimizer & Path Planner** — Sweeps TWA to maximize `boat_speed * cos(TWA)`, solves tack geometry via 2×2 ray intersection. All functions take an `interp` parameter for boat-type polymorphism.
6. **Embedded Web App** — `APP_HTML` string constant containing the full HTML/JS/CSS frontend. Canvas-based animation with controls (boat type, wind speed/direction, target position).
7. **HTTP Server** — `SimHandler` serves `GET /` (HTML) and `POST /compute` (JSON path computation). Auto-opens browser on startup.

## Boat Types

- **Laser**: Optimal VMG angle ~43-45°. Flat upwind polar but steeper than 420.
- **420**: Optimal VMG angle ~40°. Flatter upwind polar (less speed gain from bearing off). Faster on reaches due to trapeze/spinnaker. Reflects real-world experience of sailing near pinching.

## Coordinate Conventions

- Positions: (x, y) in nautical miles. x = East, y = North.
- Angles: compass bearings (0° = North, 90° = East, clockwise). Vectors: `dx = sin(bearing)`, `dy = cos(bearing)`.
- Wind direction: meteorological "FROM" convention. `wind_direction_deg=0` = wind from north.

## API

`POST /compute` with JSON body:
```json
{"boat_type": "laser", "wind_speed": 12, "wind_direction": 0, "target_x": 0, "target_y": 1}
```
Returns `{"waypoints": [...], "summary": {...}}`.

## Planned Extensions

- Time-varying wind direction (shifts) via `make_wind_fn()`
- Puffs (localized TWS increases, position-dependent)
- Start line (line segment instead of single point)
- Multiple boats / race simulation
- Downwind legs
