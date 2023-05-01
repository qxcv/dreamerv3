from dreamerv3.embodied.envs.from_gym import FromGym

class Procgen(FromGym):
    def __init__(self, task: str, **kwargs):
        # procgen doesn't need anything special, we just have this class to make
        # names shorter
        super().__init__(f'procgen:procgen-{task}-v0', **kwargs)