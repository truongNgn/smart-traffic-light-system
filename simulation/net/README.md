# SUMO network — 4-way intersection

Source-of-truth files (checked into git):

| File | Purpose |
|---|---|
| `intersection.nod.xml` | Junction nodes: center `C` (traffic light) + N/S/E/W boundaries, 100m out |
| `intersection.edg.xml` | 2-lane edges in/out of `C` for each direction |
| `vtypes.add.xml` | The 4 vehicle classes (car, motorcycle, bus, truck) |
| `generate_routes.py` | Generates demand across all 12 turning movements × 4 classes |
| `build_net.py` | Runs `netconvert` to compile the `.net.xml` |
| `intersection.sumocfg` | Ties it all together for `sumo`/`sumo-gui` |

`intersection.net.xml`, `intersection.rou.xml` and `intersection_test.rou.xml` are **generated, not committed** (see `.gitignore`) — regenerate them locally:

```bash
python -m simulation.net.build_net
python -m simulation.net.generate_routes --duration 3600 --seed 42
```

`intersection_test.rou.xml` (routes only, no background demand - used by `tests/test_grid_encoder.py`'s golden tests) is regenerated automatically by `tests/conftest.py` the first time you run `pytest`, as long as `intersection.net.xml` already exists; you don't need to generate it by hand.

Requires [SUMO](https://sumo.dlr.de/docs/Downloads.php) installed with `SUMO_HOME` set (or `sumo`/`netconvert`/`sumo-gui` on `PATH`).

Then smoke-test the TraCI wrapper:

```bash
python -m simulation.traci_wrapper.smoke_test --steps 200
# add --gui to watch it in sumo-gui
```

Editing the network visually instead of by hand: open `intersection.nod.xml` + `intersection.edg.xml` in **Netedit**, save, then re-export/re-run `build_net.py` — keep the XML source files as the thing you edit and commit, not a hand-tweaked `.net.xml`.
