# 004 Command-Conditioned RL

## Objective

This experiment tests whether the TD3 locomotion baseline selected by 003 can use a continuous `[vx_cmd, yaw_cmd]` observation to produce distinct stop, forward, left, and right behavior. TD3 is the starting algorithm for this experiment, not a universal best algorithm.

The MuJoCo model, actuator mapping, servo dynamics, safety limits, 8D action, and 33D robot-state observation are inherited from `003-RL/sesame_rl_env.py`. The only environment extension is a normalized 2D command and the fixed command-tracking reward.

## Execution order

```bash
python3 004-command-conditioned-RL/01-command-env.py
python3 004-command-conditioned-RL/02-train-td3.py
python3 004-command-conditioned-RL/03-evaluate-fixed-command.py
python3 004-command-conditioned-RL/04-command-sweep.py
python3 004-command-conditioned-RL/05-analyze-results.py
```

Before the full training run, validate the complete training pipeline with:

```bash
python3 004-command-conditioned-RL/02-train-td3.py --smoke
```

## Files

- `01-command-env.py`: 35D command-conditioned environment and G1 smoke validation.
- `02-train-td3.py`: 500,000-step TD3 training, callbacks, checkpoints, and model validation.
- `03-evaluate-fixed-command.py`: five 20-second episodes for STOP, FORWARD, LEFT, and RIGHT.
- `04-command-sweep.py`: forward-speed and yaw-rate command sweeps, five episodes per command.
- `05-analyze-results.py`: CSV analysis, figures, validity classification, and final report.
- `results/td3/`: models, monitor data, CSV files, figures, and `FINAL_REPORT.md`.

## Dependencies

The project environment must provide Python 3, NumPy, Gymnasium, MuJoCo, Stable-Baselines3, PyTorch, and Matplotlib, plus the local `sesame_ml` package used by 003.

## Experiment constants

- Algorithm: TD3
- Training steps: 500,000
- Seed: 42
- Control period: 0.02 s
- Observation: 33D robot state + 2D normalized command
- Action: 8D continuous joint targets
- Training commands: `vx ∈ [0, 0.20] m/s`, `yaw ∈ [-1, 1] rad/s`, with 15% STOP episodes
- Commands remain fixed within each episode

The baseline intentionally excludes curriculum, domain randomization, recurrent policies, command switching, reward tuning, and VLA/ROS integration.
