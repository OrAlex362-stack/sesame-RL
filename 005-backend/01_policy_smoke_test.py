from pathlib import Path
import importlib.util

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


# Load SesameCommandEnv
spec = importlib.util.spec_from_file_location("command_env", ENV_FILE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

SesameCommandEnv = module.SesameCommandEnv


# Environment
env = SesameCommandEnv(
    render_mode=None,
    stop_probability=0.0,
)

# Model
model = TD3.load(
    MODEL_FILE,
    env=env,
    device="cpu",
)

obs, info = env.reset()

env.set_command(
    vx=0.10,
    yaw=0.20,
)

print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

total_reward = 0.0

for step in range(250):

    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += float(reward)

    if step % 50 == 0:
        print(
            f"step={step:3d} "
            f"reward={reward:8.4f} "
            f"action_shape={action.shape}"
        )

    if terminated or truncated:
        print("Episode reset")

        obs, info = env.reset()

        env.set_command(
            vx=0.10,
            yaw=0.20,
        )


env.close()

print()
print("TD3 → SesameCommandEnv test PASS")
print("Total reward:", total_reward)
