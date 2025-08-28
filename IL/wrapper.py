from typing import List, Optional, Sequence, Tuple
import sys, os
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from imitation.data import rollout, types
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.state_env import PureStateEnv, StateBEVEnv


class RolloutInfoWrapper(gym.Env):
    """Add the entire episode's rewards and observations to `info` at episode end.

    Whenever done=True, `info["rollouts"]` is a dict with keys "obs" and "rews", whose
    corresponding values hold the NumPy arrays containing the raw observations and
    rewards seen during this episode.
    """

    def __init__(self, env):
        """Builds RolloutInfoWrapper.

        Args:
            env: Environment to wrap.
        """
        super().__init__(env)
        self._obs = None
        self._rews = None


    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._obs.append(types.maybe_wrap_in_dictobs(obs))
        self._rews.append(rew)

        if done:
            assert "rollout" not in info
            info["rollout"] = {
                "obs": types.stack_maybe_dictobs(self._obs),
                "rews": np.stack(self._rews),
            }
        return obs, rew, terminated, truncated, info
