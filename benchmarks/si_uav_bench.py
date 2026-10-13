"""
Benchmark + physics regression check for the Pegasus Isaac Sim 6 port (si-uav fork).

Two headless, repeatable flights:
  --mode python  Iris + Pegasus's NonlinearController on its built-in trajectory (no PX4). Records the vehicle's
                 position/attitude and every sensor's output at each physics step. With a fixed random seed, two runs
                 of the same code are bit-identical, so any change to the physics or the sensor models shows up.
  --mode px4     Pegasus drone + PX4 SITL (started once the drone rests on the floor), armed and taken off over
                 MAVLink, then hovering. Records readiness time, hover altitude/stability and the cost per step.

Both report wall-clock cost per physics step, broken down by Pegasus callback.

Usage (with ISAACSIM_PATH set):
    $ISAACSIM_PYTHON benchmarks/si_uav_bench.py --mode python --out /tmp/run.npz [--steps 3000] [--profile]
    $ISAACSIM_PYTHON benchmarks/si_uav_bench.py --mode px4 --out /tmp/px4.npz
    $ISAACSIM_PYTHON benchmarks/si_uav_bench.py --compare reference.npz candidate.npz
"""

import argparse
import functools
import os
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

PHYSICS_DT = 1.0 / 250.0

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["python", "px4"], default="python")
parser.add_argument("--out", help="where to save the recorded run (.npz)")
parser.add_argument("--steps", type=int, default=3000, help="physics steps to record in python mode (250 Hz)")
parser.add_argument("--render-hz", type=float, default=None,
                    help="step physics manually and render only at this sim-time rate (0 = never); default: let the app step and render")
parser.add_argument("--profile", action="store_true", help="print the top functions by own time (cProfile)")
parser.add_argument("--compare", nargs=2, metavar=("REFERENCE", "CANDIDATE"), help="compare two recorded runs and exit")
args = parser.parse_args()


def compare(ref_path, cand_path):
    ref, cand = np.load(ref_path), np.load(cand_path)
    n = min(len(ref["position"]), len(cand["position"]))
    dpos = np.linalg.norm(ref["position"][:n] - cand["position"][:n], axis=1)
    # Angle between attitudes: 2*acos(|<q1, q2>|)
    qa = ref["attitude"][:n] / np.linalg.norm(ref["attitude"][:n], axis=1, keepdims=True)  # float32 physics quats
    qb = cand["attitude"][:n] / np.linalg.norm(cand["attitude"][:n], axis=1, keepdims=True)  # aren't exactly unit
    dot = np.abs(np.sum(qa * qb, axis=1)).clip(0.0, 1.0)
    datt = np.degrees(2.0 * np.arccos(dot))
    print(f"compared {n} steps ({n * PHYSICS_DT:.1f} s sim)")
    print(f"  position difference: max {dpos.max() * 1000:.6f} mm, final {dpos[-1] * 1000:.6f} mm")
    print(f"  attitude difference: max {datt.max():.6f} deg, final {datt[-1]:.6f} deg")
    print(f"  distance flown (reference): {np.sum(np.linalg.norm(np.diff(ref['position'][:n], axis=0), axis=1)):.2f} m")
    for key in sorted(k for k in ref.files if k.startswith("sensor_") and k in cand.files):
        a, b = ref[key][:n], cand[key][:n]
        scale = np.abs(a).max() if np.abs(a).max() > 0 else 1.0
        print(f"  {key:24s} max abs diff {np.abs(a - b).max():.3e}  (signal max {scale:.3e})")
    print(f"  ms_per_step: reference {float(ref['ms_per_step']):.2f}, candidate {float(cand['ms_per_step']):.2f}")


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

# --- vehicle ---------------------------------------------------------------------------------------------------------
config = MultirotorConfig()
if args.mode == "python":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "utils"))
    from nonlinear_controller import NonlinearController

    config.backends = [NonlinearController(
        trajectory_file=None, results_file=os.path.join(tempfile.gettempdir(), "si_uav_bench_controller_stats.npz"),
        Ki=[0.5, 0.5, 0.5], Kr=[2.0, 2.0, 2.0])]
    robot = ROBOTS["Iris"]
else:
    from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

    config.backends = [PX4MavlinkBackend(PX4MavlinkBackendConfig({
        "vehicle_id": 0, "px4_autolaunch": False, "px4_dir": pg.px4_path,
        "px4_vehicle_model": pg.px4_default_airframe}))]
    robot = ROBOTS["Pegasus"]
vehicle = Multirotor("/World/quadrotor", robot, 0, [0.0, 0.0, 0.07],
                     Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(), config=config)

# --- per-step recording ----------------------------------------------------------------------------------------------
positions, attitudes = [], []
sensor_log = defaultdict(list)
recording = [False]


def _flatten(values):
    out = []
    for v in values:
        out.extend(np.ravel(np.asarray(v, dtype=np.float64)).tolist())
    return out


def record(dt, context=None):
    if not recording[0]:
        return
    positions.append(np.array(vehicle.state.position, dtype=np.float64))
    attitudes.append(np.array(vehicle.state.attitude, dtype=np.float64))
    for sensor in vehicle._sensors:
        state = sensor.state
        if isinstance(state, dict):
            sensor_log[sensor.sensor_type].append(_flatten(v for v in state.values() if not isinstance(v, str)))


cb = SimulationManager.register_callback(record, SimulationEvent.PHYSICS_POST_STEP)

np.random.seed(12345)  # sensor noise models use np.random; fix it so runs are comparable
timeline = omni.timeline.get_timeline_interface()
timeline.play()
simulation_app.update()  # first step(s): physics warm-up


def start_measuring():
    for k in T:
        T[k] = 0.0
    positions.clear()
    attitudes.clear()
    sensor_log.clear()
    recording[0] = True


def sim_steps():
    return len(positions)


if args.render_hz is None:
    advance = simulation_app.update  # the app steps physics (~4 steps per frame) and renders every frame
else:
    import carb

    # Stop app.update() from stepping physics; step it ourselves and render only when a frame is due
    carb.settings.get_settings().set("/app/player/playSimulations", False)
    render_every = int(round(1.0 / (args.render_hz * PHYSICS_DT))) if args.render_hz > 0 else 0
    rendered = [0]

    def advance():
        SimulationManager.step(steps=1)
        if render_every and sim_steps() % render_every == 0:
            simulation_app.update()
            rendered[0] += 1


extra = {}
if args.profile:
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

if args.mode == "python":
    start_measuring()
    t0 = time.perf_counter()
    while sim_steps() < args.steps and simulation_app.is_running():
        advance()
    wall = time.perf_counter() - t0
else:
    from pegasus.simulator.logic.backends.tools.px4_launch_tool import PX4LaunchTool
    from pymavlink import mavutil

    # Let the drone settle on the floor before PX4 starts (avoids the start-up race in Decision Log D24)
    for _ in range(100):
        simulation_app.update()
    px4 = PX4LaunchTool(pg.px4_path, 0, pg.px4_default_airframe)
    px4.launch_px4()
    mav = mavutil.mavlink_connection("udpin:127.0.0.1:14540")
    start_measuring()
    t0 = time.perf_counter()
    armed_at = takeoff_at = None
    hover = []
    last_cmd = -1e9
    timeout_steps = int(120 / PHYSICS_DT)
    while sim_steps() < timeout_steps and simulation_app.is_running():
        advance()
        t_sim = sim_steps() * PHYSICS_DT
        msg = mav.recv_match(blocking=False)
        while msg is not None:
            if msg.get_type() == "HEARTBEAT" and msg.get_srcSystem() == 1 and armed_at is None:
                if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    armed_at = t_sim
                    mav.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, float("nan"),
                                              float("nan"), float("nan"), float("nan"))
            msg = mav.recv_match(blocking=False)
        # Try to arm every 2 s of sim time until PX4 accepts (it refuses until its pre-flight checks pass)
        if armed_at is None and mav.target_system and t_sim - last_cmd > 2.0:
            mav.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            last_cmd = t_sim
        altitude = positions[-1][2] if positions else 0.0
        if armed_at is not None and takeoff_at is None and altitude > 2.0:
            takeoff_at = t_sim
        if takeoff_at is not None and t_sim > takeoff_at + 5.0:  # hover window: 5-15 s after reaching 2 m
            hover.append(positions[-1].copy())
            if t_sim > takeoff_at + 15.0:
                break
    wall = time.perf_counter() - t0
    px4.kill_px4()
    hover = np.array(hover) if hover else np.zeros((1, 3))
    extra = {"armed_at": armed_at or -1.0, "takeoff_at": takeoff_at or -1.0,
             "hover_alt_mean": hover[:, 2].mean(), "hover_alt_std": hover[:, 2].std(),
             "hover_xy_drift": np.linalg.norm(hover[-1, :2] - hover[0, :2])}
    print(f"RESULT px4: armed at {extra['armed_at']:.1f} s sim, reached 2 m at {extra['takeoff_at']:.1f} s sim, "
          f"hover altitude {extra['hover_alt_mean']:.2f} m (std {100 * extra['hover_alt_std']:.1f} cm), "
          f"xy drift over 10 s {100 * extra['hover_xy_drift']:.1f} cm")

if args.profile:
    profiler.disable()
    pstats.Stats(profiler).sort_stats("tottime").print_stats(30)

steps = sim_steps()
ms_per_step = 1000 * wall / steps
if args.render_hz is not None:
    print(f"RESULT manual stepping, rendering at {args.render_hz:g} Hz sim time ({rendered[0]} frames)")
print(f"RESULT {args.mode}: {steps} steps ({steps * PHYSICS_DT:.1f} s sim) in {wall:.1f} s wall => "
      f"{steps * PHYSICS_DT / wall:.2f}x real time, {ms_per_step:.2f} ms/step")
for k, v in T.items():
    print(f"RESULT   {k:20s} {1000 * v / steps:6.2f} ms/step")

if args.out:
    sensors = {f"sensor_{k}": np.array(v, dtype=np.float64) for k, v in sensor_log.items()}
    np.savez(args.out, position=np.array(positions), attitude=np.array(attitudes), ms_per_step=ms_per_step,
             **sensors, **extra, **{f"t_{k}": 1000 * v / steps for k, v in T.items()})
    print(f"RESULT saved {args.out}")

timeline.stop()
simulation_app.close()
