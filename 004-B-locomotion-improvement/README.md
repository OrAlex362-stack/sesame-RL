# 004-B Locomotion Improvement

## Objective

This controlled ablation tests whether narrowing only the forward tracking reward improves continuous `vx_cmd → measured vx` behavior. It compares `K_V = 25, 50, 75, 100` while keeping TD3, seed, training budget, environment, command distribution, yaw reward, dynamics, and evaluation protocol fixed.

`K_V=25` is imported from the completed 004 experiment and is not retrained. Only `K_V=50`, `75`, and `100` receive new 500,000-step runs.

## Execution order

```bash
python3 004-B-locomotion-improvement/01-reward-width-analysis.py
python3 004-B-locomotion-improvement/02-train-ablation.py
python3 004-B-locomotion-improvement/03-evaluate-ablation.py
python3 004-B-locomotion-improvement/04-command-sweep.py
python3 004-B-locomotion-improvement/05-compare-ablation.py
```

The training pipeline can be checked without contaminating formal results:

```bash
python3 004-B-locomotion-improvement/02-train-ablation.py --kv 50 --smoke
```

## Controlled variable

The only changed value is `K_V` in:

```text
2.0 * exp(-K_V * (vx - vx_cmd)^2)
```

Everything else is loaded from the unchanged 004 command environment and duplicated exactly from its TD3 configuration and evaluation protocol.

## Outputs

- `results/kv25/`: imported 004 baseline metrics and source reference.
- `results/kv50/`, `kv75/`, `kv100/`: models, callbacks, monitors, and evaluation CSV files.
- `results/figures/`: seven ablation figures.
- `results/ablation_summary.csv`: physical-metric comparison.
- `results/FINAL_REPORT.md`: Traditional Chinese scientific report.

## Optional robustness protocol

If the single-seed screen yields a clear candidate, `02-train-ablation.py` supports isolated robustness runs with `--robustness --kv ... --seed 100` or `1000`. These outputs are kept under `results/robustness/` and are not mixed into the seed-42 main ablation. The K25 seed-42 model remains the original 004 baseline.
