from ur10_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gym.envs.registration import register

register(
    id="UR10PickCube-v0",
    entry_point="ur10_sim.envs:UR10PickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="UR10PickCubeVision-v0",
    entry_point="ur10_sim.envs:UR10PickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
