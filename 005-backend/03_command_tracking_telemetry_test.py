import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from urllib.request import Request, urlopen


BASE_URL = "http://localhost:8000"

VX_CMD = 0.10
YAW_CMD = 0.00

DURATION = 5.0
SAMPLE_DT = 0.20
RESET_WAIT = 0.50

ROOT = Path.home() / "sesame-RL"
RESULT_DIR = ROOT / "005-backend" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def get_json(path):
    with urlopen(BASE_URL + path, timeout=2) as r:
        return json.loads(r.read().decode())


def post_json(path, payload=None):
    data = json.dumps(payload or {}).encode()

    req = Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(req, timeout=2) as r:
        return json.loads(r.read().decode())


print("=" * 65)
print("005-D COMMAND TRACKING TELEMETRY TEST")
print("=" * 65)

status = get_json("/status")

print("Backend :", status.get("backend"))
print("Policy  :", status.get("policy"))
print("Target  :", status.get("target_hz"), "Hz")
print(f"Command : vx={VX_CMD:.3f}, yaw={YAW_CMD:.3f}")


# --------------------------------------------------
# STOP + RESET
# --------------------------------------------------

post_json(
    "/command",
    {"vx": 0.0, "yaw": 0.0},
)

post_json("/reset")

time.sleep(RESET_WAIT)

initial = get_json("/state")

x0 = initial["base"]["position"]["x"]
y0 = initial["base"]["position"]["y"]

episode0 = initial["sim"]["episode"]

deadline0 = initial["controller"]["deadline_misses"]


# --------------------------------------------------
# COMMAND
# --------------------------------------------------

post_json(
    "/command",
    {
        "vx": VX_CMD,
        "yaw": YAW_CMD,
    },
)

samples = []

start_wall = time.perf_counter()

while time.perf_counter() - start_wall < DURATION:

    time.sleep(SAMPLE_DT)

    s = get_json("/state")

    action = s["policy"]["action"]

    samples.append({
        "wall_time":
            time.perf_counter() - start_wall,

        "sim_time":
            s["sim"]["time"],

        "episode":
            s["sim"]["episode"],

        "cmd_vx":
            s["command"]["vx"],

        "cmd_yaw":
            s["command"]["yaw"],

        "x":
            s["base"]["position"]["x"],

        "y":
            s["base"]["position"]["y"],

        "z":
            s["base"]["position"]["z"],

        "roll":
            s["base"]["orientation"]["roll"],

        "pitch":
            s["base"]["orientation"]["pitch"],

        "yaw":
            s["base"]["orientation"]["yaw"],

        "vx":
            s["base"]["velocity"]["linear"]["x"],

        "vy":
            s["base"]["velocity"]["linear"]["y"],

        "vz":
            s["base"]["velocity"]["linear"]["z"],

        "wx":
            s["base"]["velocity"]["angular"]["x"],

        "wy":
            s["base"]["velocity"]["angular"]["y"],

        "wz":
            s["base"]["velocity"]["angular"]["z"],

        "reward":
            s["sim"]["reward"],

        "hz":
            s["controller"]["actual_hz"],

        "compute_ms":
            s["controller"]["compute_ms"],

        "deadline_misses":
            s["controller"]["deadline_misses"],

        **{
            f"action_{i}": float(v)
            for i, v in enumerate(action)
        },
    })


# --------------------------------------------------
# STOP
# --------------------------------------------------

post_json(
    "/command",
    {"vx": 0.0, "yaw": 0.0},
)


# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_file = (
    RESULT_DIR
    / f"005D_tracking_vx_{VX_CMD:.2f}_yaw_{YAW_CMD:.2f}_{timestamp}.csv"
)

with csv_file.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=samples[0].keys(),
    )

    writer.writeheader()
    writer.writerows(samples)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

last = samples[-1]

dx = last["x"] - x0
dy = last["y"] - y0

displacement = math.sqrt(dx**2 + dy**2)

vx_values = [s["vx"] for s in samples]

half = len(samples) // 2
steady = samples[half:]

steady_vx = mean(
    s["vx"] for s in steady
)

steady_error = abs(
    VX_CMD - steady_vx
)

mae_vx = mean(
    abs(VX_CMD - s["vx"])
    for s in samples
)

mean_hz = mean(
    s["hz"]
    for s in samples
)

max_abs_vy = max(
    abs(s["vy"])
    for s in samples
)

max_abs_wz = max(
    abs(s["wz"])
    for s in samples
)

episode_changes = (
    last["episode"] - episode0
)

deadline_delta = (
    last["deadline_misses"] - deadline0
)


# Action saturation
action_values = []

for s in samples:
    for i in range(8):
        action_values.append(
            abs(s[f"action_{i}"])
        )

saturated = sum(
    v >= 0.99
    for v in action_values
)

saturation_ratio = (
    saturated / len(action_values)
)


# --------------------------------------------------
# REPORT
# --------------------------------------------------

print("\n=== TELEMETRY INTEGRITY ===")

print(f"Samples           : {len(samples)}")
print(f"Mean controller Hz: {mean_hz:.3f}")
print(f"Episode changes   : {episode_changes}")
print(f"Deadline misses   : {deadline_delta}")


print("\n=== POSITION ===")

print(f"ΔX           : {dx:.6f} m")
print(f"ΔY           : {dy:.6f} m")
print(f"Displacement : {displacement:.6f} m")


print("\n=== COMMAND TRACKING ===")

print(f"Command vx          : {VX_CMD:.6f} m/s")
print(f"Mean steady-state vx: {steady_vx:.6f} m/s")
print(f"Steady-state error  : {steady_error:.6f} m/s")
print(f"Full-run vx MAE     : {mae_vx:.6f} m/s")

print(f"Max |vy|            : {max_abs_vy:.6f} m/s")
print(f"Max |wz|            : {max_abs_wz:.6f} rad/s")


print("\n=== POLICY ===")

print(
    f"Action saturation : "
    f"{saturation_ratio * 100:.2f}%"
)


print("\n=== 005-D RESULT ===")

telemetry_ok = (
    len(samples) > 0
    and 45.0 <= mean_hz <= 55.0
)

if telemetry_ok:
    print("Telemetry infrastructure : PASS")
else:
    print("Telemetry infrastructure : FAIL")

print("Command tracking         : REPORT ONLY")
print(
    "(tracking quality belongs to 004 / 004-B)"
)

print("\nCSV:")
print(csv_file)

print("=" * 65)