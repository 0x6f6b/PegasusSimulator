"""
Benchmark + physics regression check for the Pegasus Isaac Sim 6 port (si-uav fork).

Flies a fixed, repeatable flight headless and records:
  - wall-clock cost per physics step, broken down by Pegasus callback
  - the vehicle's position and attitude at every physics step (to compare against a reference run)

Usage (with ISAACSIM_PATH set):
    $ISAACSIM_PYTHON benchmarks/si_uav_bench.py --out /tmp/run.npz [--steps 3000]
    $ISAACSIM_PYTHON benchmarks/si_uav_bench.py --compare reference.npz candidate.npz

The flight uses Pegasus's NonlinearController (no PX4) on the Iris following its built-in trajectory, on the
Flat Plane environment. The controller uses the true vehicle state and simulated time only, so two runs of the
same code should produce the same trajectory.
"""

import argparse
import functools
import os
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out", help="where to save the recorded run (.npz)")
parser.add_argument("--steps", type=int, default=3000, help="physics steps to record (250 Hz)")
parser.add_argument("--compare", nargs=2, metavar=("REFERENCE", "CANDIDATE"), help="compare two recorded runs and exit")
args = parser.parse_args()


def compare(ref_path, cand_path):
    ref, cand = np.load(ref_path), np.load(cand_path)
    n = min(len(ref["position"]), len(cand["position"]))
    dpos = np.linalg.norm(ref["position"][:n] - cand["position"][:n], axis=1)
    # Angle between attitudes: 2*acos(|<q1, q2>|)
    dot = np.abs(np.sum(ref["attitude"][:n] * cand["attitude"][:n], axis=1)).clip(0.0, 1.0)
    datt = np.degrees(2.0 * np.arccos(dot))
    print(f"compared {n} steps ({n * 0.004:.1f} s sim)")
    print(f"  position difference: max {dpos.max() * 1000:.3f} mm, final {dpos[-1] * 1000:.3f} mm")
    print(f"  attitude difference: max {datt.max():.4f} deg, final {datt[-1]:.4f} deg")
    print(f"  distance flown (reference): {np.sum(np.linalg.norm(np.diff(ref['position'][:n], axis=0), axis=1)):.2f} m")
    for name in ("ms_per_step",):
        print(f"  {name}: reference {float(ref[name]):.2f}, candidate {float(cand[name]):.2f}")


if args.compare:
    compare(*args.compare)
    sys.exit(0)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.timeline
import omni.usd
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.vehicles.vehicle import Vehicle
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "utils"))
from nonlinear_controller import NonlinearController

# --- per-callback timing -------------------------------------------------------------------------------------------
T = defaultdict(float)


def timed(cls, name, label):
    orig = getattr(cls, name)

    @functools.wraps(orig)  # SimulationManager looks callbacks up by name
    def wrapper(self, *a, **k):
        s = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            T[label] += time.perf_counter() - s

    setattr(cls, name, wrapper)


timed(Vehicle, "update_state", "update_state")
timed(Multirotor, "update", "multirotor.update")
timed(Vehicle, "update_sensors", "update_sensors")

# --- scene: load synchronously and wait until it is fully loaded (avoids the spawn-before-floor race) --------------
pg = PegasusInterface()
pg.initialize_world()
pg.load_asset(SIMULATION_ENVIRONMENTS["Flat Plane"], "/World/layout")
ctx = omni.usd.get_context()
settled = 0
while settled < 30:
    simulation_app.update()
    _, loaded, total = ctx.get_stage_loading_status()
    settled = settled + 1 if loaded >= total else 0

# --- vehicle: Iris + Pegasus's nonlinear controller on its built-in trajectory -------------------------------------
config = MultirotorConfig()
config.backends = [NonlinearController(trajectory_file=None, results_file=os.path.join(tempfile.gettempdir(), "si_uav_bench_controller_stats.npz"), Ki=[0.5, 0.5, 0.5], Kr=[2.0, 2.0, 2.0])]
vehicle = Multirotor("/World/quadrotor", ROBOTS["Iris"], 0, [0.0, 0.0, 0.07],
                     Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(), config=config)

positions, attitudes = [], []


def record(dt, context=None):
    if len(positions) < args.steps:
        positions.append(np.array(vehicle.state.position, dtype=np.float64))
        attitudes.append(np.array(vehicle.state.attitude, dtype=np.float64))


cb = SimulationManager.register_callback(record, SimulationEvent.PHYSICS_POST_STEP)

timeline = omni.timeline.get_timeline_interface()
timeline.play()
simulation_app.update()  # first step(s): physics warm-up
for k in T:
    T[k] = 0.0
positions.clear()
attitudes.clear()

t0 = time.perf_counter()
while len(positions) < args.steps and simulation_app.is_running():
    simulation_app.update()
wall = time.perf_counter() - t0
steps = len(positions)

ms_per_step = 1000 * wall / steps
print(f"RESULT {steps} steps ({steps * 0.004:.1f} s sim) in {wall:.1f} s wall => {steps * 0.004 / wall:.2f}x real time, "
      f"{ms_per_step:.2f} ms/step")
for k, v in T.items():
    print(f"RESULT   {k:20s} {1000 * v / steps:6.2f} ms/step")

if args.out:
    np.savez(args.out, position=np.array(positions), attitude=np.array(attitudes), ms_per_step=ms_per_step,
             **{f"t_{k}": 1000 * v / steps for k, v in T.items()})
    print(f"RESULT saved {args.out}")

timeline.stop()
simulation_app.close()
