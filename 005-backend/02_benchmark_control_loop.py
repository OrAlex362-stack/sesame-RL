from pathlib import Path
import importlib.util
import time
import numpy as np

from stable_baselines3 import TD3


ROOT = Path.home() / "sesame-RL"

ENV_FILE = ROOT / "004-command-conditioned-RL" / "01-command-env.py"
MODEL_FILE = (
    ROOT
    / "004-command-conditioned-RL"
    / "results"
    / "td3"
    / "best_model"
    / "best_model.zip"
)

VX = 0.10
YAW = 0.20

WARMUP = 500
N_UNCAPPED = 3000
N_REALTIME = 1000

TARGET_DT = 0.020
TARGET_HZ = 1.0 / TARGET_DT


# --------------------------------------------------
# Load environment
# --------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "command_env",
    ENV_FILE,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

SesameCommandEnv = module.SesameCommandEnv


env = SesameCommandEnv(
    render_mode=None,
    stop_probability=0.0,
)

model = TD3.load(
    MODEL_FILE,
    env=env,
    device="cpu",
)

obs, info = env.reset()
env.set_command(VX, YAW)


def reset_if_needed(obs, terminated, truncated):
    if terminated or truncated:
        obs, info = env.reset()
        env.set_command(VX, YAW)

    return obs


def percentile_report(name, values_ms):
    x = np.asarray(values_ms)

    print(f"\n{name}")
    print("-" * 55)
    print(f"mean   : {x.mean():8.4f} ms")
    print(f"median : {np.median(x):8.4f} ms")
    print(f"p95    : {np.percentile(x, 95):8.4f} ms")
    print(f"p99    : {np.percentile(x, 99):8.4f} ms")
    print(f"max    : {x.max():8.4f} ms")


# --------------------------------------------------
# Warmup
# --------------------------------------------------

print("=== Sesame RL Control Benchmark ===")
print("Model :", MODEL_FILE)
print("Device: CPU")
print("Target:", TARGET_HZ, "Hz")
print("Command:", VX, YAW)

print("\nWarmup...")

for _ in range(WARMUP):

    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    obs, reward, terminated, truncated, info = env.step(action)

    obs = reset_if_needed(
        obs,
        terminated,
        truncated,
    )


# --------------------------------------------------
# Test A: uncapped
# --------------------------------------------------

print("\n=== TEST A: UNCAPPED THROUGHPUT ===")

predict_ms = []
step_ms = []
total_ms = []

for _ in range(N_UNCAPPED):

    t0 = time.perf_counter()

    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    t1 = time.perf_counter()

    obs, reward, terminated, truncated, info = env.step(action)

    t2 = time.perf_counter()

    predict_ms.append((t1 - t0) * 1000)
    step_ms.append((t2 - t1) * 1000)
    total_ms.append((t2 - t0) * 1000)

    obs = reset_if_needed(
        obs,
        terminated,
        truncated,
    )


percentile_report("TD3.predict()", predict_ms)
percentile_report("MuJoCo env.step()", step_ms)
percentile_report("Total compute", total_ms)

mean_compute = np.mean(total_ms) / 1000.0

print("\nMaximum compute throughput")
print("-" * 55)
print(f"Mean compute time : {mean_compute * 1000:.4f} ms")
print(f"Max theoretical Hz: {1.0 / mean_compute:.2f} Hz")


# --------------------------------------------------
# Test B: actual 50 Hz scheduling
# --------------------------------------------------

print("\n=== TEST B: 50 Hz REAL-TIME LOOP ===")

period_ms = []
compute_rt_ms = []
lateness_ms = []

deadline_misses = 0

next_tick = time.perf_counter()
previous_start = None


for _ in range(N_REALTIME):

    scheduled = next_tick

    now = time.perf_counter()

    if now < scheduled:
        time.sleep(scheduled - now)

    start = time.perf_counter()

    if previous_start is not None:
        period_ms.append(
            (start - previous_start) * 1000
        )

    previous_start = start

    lateness_ms.append(
        max(0.0, start - scheduled) * 1000
    )


    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    obs, reward, terminated, truncated, info = env.step(action)

    end = time.perf_counter()

    compute_rt_ms.append(
        (end - start) * 1000
    )


    if end > scheduled + TARGET_DT:
        deadline_misses += 1


    obs = reset_if_needed(
        obs,
        terminated,
        truncated,
    )

    next_tick = scheduled + TARGET_DT


period = np.asarray(period_ms)

actual_hz = 1.0 / (period.mean() / 1000.0)

percentile_report(
    "Actual control period",
    period_ms,
)

percentile_report(
    "Realtime compute",
    compute_rt_ms,
)

percentile_report(
    "Scheduler lateness",
    lateness_ms,
)


print("\n=== REAL-TIME RESULT ===")
print(f"Target Hz        : {TARGET_HZ:.2f}")
print(f"Actual mean Hz   : {actual_hz:.2f}")
print(f"Period std       : {period.std():.4f} ms")
print(
    f"Deadline misses  : "
    f"{deadline_misses}/{N_REALTIME} "
    f"({deadline_misses / N_REALTIME * 100:.2f}%)"
)

env.close()