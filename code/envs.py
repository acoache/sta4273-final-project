import numpy as np
import gym
import collections
from typing import Tuple

"""
# Environment functions

* DeepSea environment
* CartPole environment
* Perturb_DeepSea -- function to add non-stationarity in the DeepSea environment
* Perturb_CartPole -- function to add non-stationarity in the CartPole environment
"""

# stores the game state of a round
TimeStep = collections.namedtuple('TimeStep', ['observation' ,'reward', 'continue_play'])

class DeepSea(object):

    def __init__(self,
                size: int, 
                seed: int = None,
                prob: int = 0.5,
                randomize: bool = True):
        self._size = size # number of states in the game
        self._move_cost = 0.01 / size # cost to move to the right
        self._goal_reward = 1. # reward for reaching the bottom right corner

        # initialize state
        self._column = 0
        self._row = 0

        if randomize:
            # randomize whether left or right is encoded by a 0  
            rng = np.random.RandomState(seed)
            self._action_mapping = rng.binomial(1, prob, size)
        else:
            # 0 is left and 1 is right 
            self._action_mapping = np.ones(size)

        self._reset_next_step = False

    def step(self, action: int) -> TimeStep:
        if self._reset_next_step:
            return self.reset()
        # Remap actions according to column (action_right = go right)
        action_right = action == self._action_mapping[self._column]

        # Compute the reward
        reward = 0.
        if self._column == self._size-1 and action_right:
            reward += self._goal_reward

        # State dynamics
        if action_right:  # right
            self._column = np.clip(self._column + 1, 0, self._size-1)
            reward -= self._move_cost
        else:  # left
            self._column = np.clip(self._column - 1, 0, self._size-1)

        # Compute the observation
        self._row += 1
        if self._row == self._size:
            observation = self._get_observation(self._row-1, self._column)
            self._reset_next_step = True
            return TimeStep(reward=reward, observation=observation, continue_play=0.)
        else:
            observation = self._get_observation(self._row, self._column)
            return TimeStep(reward=reward, observation=observation, continue_play=1.)

    def reset(self) -> TimeStep:
        self._reset_next_step = False
        self._column = 0
        self._row = 0
        observation = self._get_observation(self._row, self._column)

        return TimeStep(reward=None, observation=observation, continue_play=1.)

    def _get_observation(self, row, column) -> np.ndarray:
        observation = np.zeros(shape=(self._size, self._size), dtype=np.float32)
        observation[row, column] = 1

        return observation

    # helper for what the game size is
    def obs_shape(self) -> Tuple[int]:
        return self.reset().observation.shape

    # helper for what the number of actions is
    def num_actions(self) -> int:
        return 2
    # helper for what the optimal reward is
    def optimal_reward(self) -> float:
        return self._goal_reward - self._move_cost * self._size

# this is a wrapper so that CartPole can be treated like DeepSea
class CartPole(object):

    def __init__(self,
               gravity = 9.8,
               half_length = 0.5):

        self.env = gym.make('CartPole-v0')
        self.env.gravity = gravity
        self.env.length= half_length # from gym: actually half the pole's length
        self.obs_shape = self.env.observation_space.shape[0] + 2

    def step(self, action: int) -> TimeStep:

        # Execute the action
        current_state = self.env.step(action)

        # Compute the reward
        reward = current_state[1]

        # Compute the observation
        observation = np.concatenate((current_state[0], [self.env.gravity, self.env.length]))

        # Compute the continue_play variable
        continue_play = 1. - current_state[2]

        return TimeStep(reward=reward, observation=observation, continue_play=continue_play)

    def reset(self) -> TimeStep:
        observation = np.concatenate((self.env.reset(), [self.env.gravity, self.env.length]))
        return TimeStep(reward=None, observation=observation, continue_play=1.)
    
    # helper for what the game size is
    def obs_shape(self) -> Tuple[int]:
        return self.obs_shape
    
    # helper for what the number of actions is
    def num_actions(self) -> int:
        return 2
    
    # helper for what the optimal reward is
    def optimal_reward(self) -> float:
        return 200.

# change the board slightly after each episode
def Perturb_DeepSea(DeepSeaEnv,
                    noisy_prob=0.1):
    # flip a biased coin to decide whether to change the board
    if np.random.binomial(n=1,p=noisy_prob,size=1):
        # uniformly select a column to flip
        col_flip = np.random.randint(0,DeepSeaEnv._size)
        DeepSeaEnv._action_mapping[col_flip] = DeepSeaEnv._action_mapping[col_flip] == 0

# noisy drift in physic parameters of CartPole
def Perturb_CartPole(CartPoleEnv,
                     perturb_gravity = False,
                     perturb_length = False,
                     gravity_drift = 0.95,
                     gravity_vol = 0.3,
                     length_drift = 0.9,
                     length_vol = 0.1):
    if perturb_gravity:
        CartPoleEnv.env.gravity = max(1.0, CartPoleEnv.env.gravity*gravity_drift + gravity_vol*np.random.randn())
    if perturb_length:
        CartPoleEnv.env.length = max(0.2, CartPoleEnv.env.length*length_drift + length_vol*np.random.randn())