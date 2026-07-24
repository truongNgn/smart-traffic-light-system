# State encoding: SUMO → 80-cell grid

Implements the state space from Sahal et al. (2023): each 100m approach
segment discretized into cells, 4 approaches → 80 total, fed directly into
the DQN's input layer.

## Layout

```
common.constants.GRID_CELLS_TOTAL          = 80
common.constants.GRID_CELLS_PER_APPROACH   = 20   (80 / 4 directions)
common.constants.CELL_LENGTH_M             = 5.0  (100m / 20 cells)
```

The 80-vector is 4 contiguous blocks of 20, ordered by
`common.constants.Direction`'s integer value:

| Block | Indices | Direction | Incoming edge (simulation/net/intersection.edg.xml) |
|---|---|---|---|
| 0 | `grid[0:20]`  | `EAST`  (0) | `E2C` |
| 1 | `grid[20:40]` | `NORTH` (1) | `N2C` |
| 2 | `grid[40:60]` | `WEST`  (2) | `W2C` |
| 3 | `grid[60:80]` | `SOUTH` (3) | `S2C` |

Within a block, **cell 0 is nearest the stop line** (the junction) and
**cell 19 is nearest the network boundary**, 100m out.

```
stop line                                              network edge
  |cell0|cell1|cell2| ... |cell17|cell18|cell19|
  |<-5m->|                                     |<-100m total->|
```

## Cell value

Each cell is **binary occupancy**: `1.0` if at least one vehicle (in either
of the approach's 2 lanes) currently sits in that 5m band, else `0.0`. Lanes
are collapsed - the grid has no separate lane dimension, matching the
paper's single 80-neuron input layer (not 80 x num_lanes).

## Computing a cell index

For a vehicle on incoming edge `E` with `lanePosition` (SUMO's distance from
the start of the edge, via `traci.vehicle.getLanePosition`):

```
distance_to_stop_line = edge_length_m - lane_position
cell_in_approach      = floor(distance_to_stop_line / CELL_LENGTH_M)
cell_in_approach       = clamp(cell_in_approach, 0, 19)   # vehicles beyond
                                                            # 100m still land
                                                            # in the last cell
global_index = direction.value * 20 + cell_in_approach
```

`edge_length_m` here is **not** the `APPROACH_LENGTH_M` constant (100) taken
literally - `netconvert` trims each edge to fit the junction's actual shape
(turn radii, lane widths), so real edge lengths end up a few meters short of
100 and differ slightly per direction. `GridEncoder` queries the real length
via `traci.lane.getLength(f"{edge_id}_0")` the first time it sees an edge and
caches it, instead of trusting the nominal constant - using the constant
directly caused an off-by-one-cell bug caught by
`tests/test_grid_encoder.py`'s golden tests.

Vehicles not currently on one of the 4 approach edges (i.e. already inside
the junction's internal lanes, or already past it on an outgoing edge) are
excluded from the grid - the paper's grid only represents *approaching*
traffic.

## Worked example

A car on edge `N2C` (100m long) with `lanePosition = 92.0`:

```
distance_to_stop_line = 100.0 - 92.0 = 8.0
cell_in_approach      = floor(8.0 / 5.0) = 1
global_index          = Direction.NORTH.value * 20 + 1 = 1*20 + 1 = 21
```

So `grid[21] = 1.0` - the second cell of the North block, i.e. this car is
5-10m from the stop line. This exact case is covered by
`tests/test_grid_encoder.py::test_vehicle_at_known_offset_lands_in_expected_cell`.

## Waiting time

`simulation/state/waiting_time.py` sums `traci.vehicle.getWaitingTime(id)`
across vehicles. SUMO's own definition of that metric already matches the
paper's threshold (`common.constants.STOPPED_SPEED_THRESHOLD_MPS = 0.1`
m/s) - it accumulates time since the vehicle's speed last exceeded 0.1 m/s -
so no separate speed-threshold bookkeeping is implemented; the reward engine
(Stage 3) diffs this value step-to-step: `r_t = t^T_{t-1} - t^T_t`.

## Extending to a different network

If the approach lengths ever differ from the paper's 100m, `GridEncoder`
accepts an `edge_lengths_m` override rather than assuming a constant - pull
real lengths from `traci.lane.getLength()` at startup rather than hardcoding
them a second time.
