import os
import gymnasium as gym
import gym_pusht # required to register the environment, even if not directly used in this file
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors

# Create the environment with pixel observations and rendering enabled
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array_list")
observation, info = env.reset()
frames = []
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pretrained diffusion policy
model_ref = "./my_diffusion_pusht/checkpoints/last/pretrained_model" if os.path.exists("./my_diffusion_pusht/checkpoints/last/pretrained_model") else "pbelevich/diffusion_pusht"
policy = DiffusionPolicy.from_pretrained(model_ref)
policy.to(device)
policy.eval()
policy.reset()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=model_ref,
)

# Run the policy for a fixed number of steps, capturing rendered frames
for _ in tqdm(range(1000)):
    policy_observation = preprocess_observation(observation)
    policy_observation = preprocessor(policy_observation)
    with torch.inference_mode():
        action_tensor = policy.select_action(policy_observation)
    action_tensor = postprocessor(action_tensor)
    action = action_tensor.detach().to("cpu").numpy().astype(np.float32)[0]
    action = np.clip(action, env.action_space.low, env.action_space.high)
    observation, reward, terminated, truncated, info = env.step(action)
    images = env.render()
    if images is not None:
        if isinstance(images, list):
            frames.extend(images)
        else:
            frames.append(images)

    if terminated or truncated:
        observation, info = env.reset()
        policy.reset()

env.close()

# Save frames to MP4 or GIF
if frames:
    fps = env.metadata.get("render_fps", 30)
    try:
        mp4_path = "pusht_rollout.mp4"
        with imageio.get_writer(mp4_path, format="FFMPEG", fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Saved {len(frames)} frames to {mp4_path} at {fps} FPS")
    except Exception as exc:
        print(f"MP4 export failed ({exc}); attempting GIF export instead")
        gif_path = "pusht_rollout.gif"
        duration = 1.0 / fps if fps else 1.0 / 30.0
        imageio.mimsave(gif_path, frames, duration=duration)
        print(
            f"MP4 export failed ({exc}); saved {len(frames)} frames to {gif_path} instead"
        )
else:
    print("No frames captured; video not written.")
