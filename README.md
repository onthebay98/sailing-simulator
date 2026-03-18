# Sailing Simulator

**[Try it live](https://sailing-simulator-alpha.vercel.app)**

An interactive browser-based sailing simulator that computes and animates the fastest path around an upwind-downwind racecourse through spatially-varying wind. Built from real dinghy and keelboat polar speed data with grid-based Dijkstra weather routing.

## What It Does

A sailboat can't sail directly into or straight downwind efficiently — it must zigzag (tacking upwind, jibing downwind) at angles that maximize its progress toward the target. This simulator solves that optimization problem and visualizes the result.

Given a course layout (start, upwind mark, finish) and wind conditions, it:

1. **Generates a spatially-varying wind field** — wind speed and direction vary smoothly across the course area using sinusoidal perturbations, creating realistic conditions where the optimal path isn't just a simple zigzag
2. **Finds the optimal path** via grid-based Dijkstra weather routing, evaluating ~10,000 tack-state-aware graph nodes with realistic maneuver penalties
3. **Lets you race against the computer** — in Challenge mode, place multiple tack and jibe waypoints, then watch your boat race the optimal ghost boat in real time
4. **Animates the boats** traversing the course, showing speed, heading, point of sail, and tack/jibe maneuvers

## Technical Highlights

- **Weather routing**: Grid-based Dijkstra over a tack-state-extended graph `(x, y, tack)` — properly penalizes maneuvers with boat-specific tack/jibe times
- **Spatially-varying wind**: `WindField` class with smooth sinusoidal perturbations seeded deterministically from course parameters
- **Polar speed interpolation**: scipy `RegularGridInterpolator` over TWS × TWA grids with realistic no-go zones
- **Vectorized graph construction**: numpy-based edge cost computation for fast pathfinding
- **Three boat models** with differentiated polar data and maneuver penalties:
  - **Laser** — moderate upwind angle (~44.5°), singlehander. Tack: 6s, jibe: 5s
  - **420** — tighter upwind angle (~40°), faster downwind with spinnaker. Tack: 8s, jibe: 10s
  - **J/24** — wider upwind angle (~48°), heavier displacement keelboat. Tack: 12s, jibe: 10s
- **Single-file architecture** — Python backend with embedded HTML5 Canvas frontend, no build step, no external JS dependencies
- **JSON API** — `POST /compute` accepts course parameters, returns waypoints, wind grid, and summary statistics

## Setup

Requires Python 3.10+ with numpy and scipy:

```bash
pip install numpy scipy
```

Run:

```bash
python3 sailing_sim.py
```

This starts a local server and opens the simulator in your browser at `http://localhost:8420`.

## Usage

### Simulate Mode

- **Boat**: Select Laser, 420, or J/24 to see how hull characteristics change optimal strategy
- **Wind Speed / Direction**: Adjust base wind conditions — the spatial wind field varies around these values
- **Start / Mark / Finish**: Slide to reposition marks and create asymmetric courses
- **Laylines**: Toggle to show optimal tack/jibe angle lines from mark
- Wind arrows on the course show the local wind field

### Challenge Mode

Test your tactical decision-making against the computer's optimal path:

1. A random course is generated with spatially-varying wind
2. **Click** the canvas to place tack waypoints (upwind leg)
3. Click **Downwind** button to switch to jibe placement (downwind leg)
4. Click **Undo** (or press **Backspace**) to remove the last placed waypoint
5. Hit **Race** to animate both boats and see your tactics score

Score = `optimal_time / your_time × 100%`. The wind arrows on the course help you read the conditions and plan your route.

The info panel (left) shows live telemetry. The summary panel (right) shows optimal angles, VMG, tack/jibe counts, and total time.

## Deployment

Deployed on Vercel with separate files:
- `api/compute.py` — serverless function (weather routing backend)
- `public/index.html` — frontend

These must be kept in sync with `sailing_sim.py`.

## Background

I crewed 420s competitively (my skipper won youth worlds twice). This project grew out of wanting to quantify the upwind VMG tradeoffs we debated on the water — particularly why 420 sailors pinch more aggressively than Laser sailors, and whether that's actually optimal. The polar data and resulting optimal angles match real-world experience.

The spatially-varying wind and weather routing make the challenge mode genuinely tactical — you need to read the wind field and decide not just where to tack, but how many tacks to take and whether to sail into stronger wind or take a shorter path through lighter air.
