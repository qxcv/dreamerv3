from typing import cast
import gymnasium
from minigrid.wrappers import FullyObsWrapper, ObservationWrapper
from dreamerv3.embodied.envs.from_gymnasium import FromGymnasium


class HideMission(ObservationWrapper):
    """Remove the 'mission' string from the observation."""
    def __init__(self, env):
        super().__init__(env)
        obs_space = cast(gymnasium.spaces.Dict, self.observation_space)
        obs_space.spaces.pop('mission')

    def observation(self, observation: dict):
        observation.pop('mission')
        return observation


class Minigrid(FromGymnasium):
    def __init__(self, task: str, fully_observable: bool, hide_mission: bool):
        env = gymnasium.make(f"MiniGrid-{task}-v0", render_mode="rgb_array")
        if fully_observable:
            env = FullyObsWrapper(env)
        if hide_mission:
            env = HideMission(env)
        super().__init__(env=env)