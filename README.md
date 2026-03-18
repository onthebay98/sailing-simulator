# Sailing Simulator

**[Try it live](https://sailing-simulator-alpha.vercel.app)**

An interactive browser-based sailing simulator that computes and animates the fastest path around an upwind-downwind racecourse. Built from real dinghy and keelboat polar speed data and first-principles VMG (Velocity Made Good) optimization.

## What It Does

A sailboat can't sail directly into or straight downwind efficiently — it must zigzag (tacking upwind, jibing downwind) at angles that maximize its progress toward the target. This simulator solves that optimization problem geometrically and visualizes the result.

Given a course layout (start, upwind mark, finish) and wind conditions, it:

1. **Finds the optimal sailing angle** by sweeping True Wind Angle (TWA) against realistic polar speed curves and maximizing VMG — the velocity component toward the destination
2. **Computes the fastest path** via ray intersection geometry, determining where to tack or jibe
3. **Compares single-leg vs. two-leg paths** and selects whichever is faster — important for downwind legs where dead runs are reachable but slower than jibing at a broader angle
4. **Animates the boat** traversing the computed path in real time, showing speed, heading, point of sail, and tack/jibe maneuvers

## Technical Highlights

- **Optimization**: VMG maximization over interpolated polar surfaces (scipy `RegularGridInterpolator` over TWS x TWA grids) — upwind and downwind each have distinct optimal angles
- **Geometry**: Tack/jibe points solved as 2x2 linear systems (ray intersection); the same generalized solver handles both upwind tacking and downwind jibing
- **Three boat models** with differentiated polar data reflecting real-world performance:
  - **Laser** — moderate upwind angle (~44.5°), standard singlehander
  - **420** — tighter upwind angle (~40°) due to flatter polar, faster downwind with spinnaker
  - **J/24** — wider upwind angle (~48°), heavier displacement keelboat
- **Single-file architecture** (~1100 lines) — Python backend with embedded HTML5 Canvas frontend, no build step, no external JS dependencies
- **JSON API** — `POST /compute` accepts course parameters, returns waypoints and summary statistics

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

- **Boat**: Select Laser, 420, or J/24 to see how hull characteristics change optimal strategy
- **Wind Direction**: Rotate the wind — the optimal path reconfigures in response
- **Start / Mark / Finish**: Slide to reposition marks and create asymmetric courses
- **Simulate**: Runs the animation; pause, resume, or reset at any time

The info panel (left) shows live telemetry. The summary panel (right) shows optimal angles, VMG, tack/jibe counts, and total time for each leg.

## Background

I crewed 420s competitively (my skipper won youth worlds twice). This project grew out of wanting to quantify the upwind VMG tradeoffs we debated on the water — particularly why 420 sailors pinch more aggressively than Laser sailors, and whether that's actually optimal. The polar data and resulting optimal angles match real-world experience.
