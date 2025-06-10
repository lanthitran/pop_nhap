#!/usr/bin/env python3

import gym
import numpy as np
import grid2op
from grid2op.gym_compat import GymEnv
from learn2learn.gym.envs.meta_env import MetaEnv
from RL2Grid.env.utils import make_env, make_env_for_gym

class Grid2OpDirectionEnv(MetaEnv, gym.Env):
    """
    **Description**

    This environment requires the agent to learn to control a power grid in different scenarios.
    At each time step the agent receives a signal composed of a control cost and a reward
    based on the grid's performance. The tasks are different grid scenarios with varying
    load profiles, line failures, or weather conditions. The target is to maintain grid
    stability while minimizing line overloads and maximizing power flow efficiency.

    **Credit**

    Adapted from Grid2Op implementation and learn2learn's AntDirectionEnv.

    **References**

    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." arXiv [cs.LG].
    2. Rothfuss et al. 2018. "ProMP: Proximal Meta-Policy Search." arXiv [cs.LG].
    3. Grid2Op Documentation and Implementation.
    """

    def __init__(self, task=None, env_name="l2rpn_case14_sandbox", action_type="topology"):
        MetaEnv.__init__(self, task)
        gym.Env.__init__(self)
        
        # Create arguments for make_env
        class Args:
            def __init__(self):
                self.env_id = env_name
                self.action_type = action_type
                self.env_config_path = "scenario.json"
                self.norm_obs = True
                self.use_heuristic = False
                self.seed = 42
        
        args = Args()
        
        # Create environment using make_env
        env_creator = make_env(args, 0) # or make_env_for_gym ?
        self.env = env_creator()  # Keep this for accessing Grid2Op-specific features
        
        # Create Gym environment
        self.gym_env = GymEnv(self.env)
        
        # Set observation and action spaces
        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space
        
        # Store current chronic ID
        self.current_chronic_id = None
        
        # Initialize task parameters
        self.task = task if task is not None else self.sample_tasks(1)[0]
        self.set_task(self.task)

    # -------- MetaEnv Methods --------
    def set_task(self, task):
        """Set the current task configuration."""
        MetaEnv.set_task(self, task)
        
        # Update environment parameters based on task
        if 'chronics_id' in task:
            # Store the chronic ID
            self.current_chronic_id = task['chronics_id']
            # Reset environment with the specified chronic ID
            self.gym_env.reset(options={"time serie id": self.current_chronic_id})
            
        if 'weather_conditions' in task:
            # TODO 
            # Note: Weather conditions need to be set on the base Grid2Op environment
            # as it's not exposed through the Gym interface
            self.env.set_weather_conditions(task['weather_conditions'])  # not work yet

    def sample_tasks(self, num_tasks):
        """Sample different Grid2Op scenarios as tasks."""
        tasks = []
        # Need to access chronics_handler through the base environment... or not?
        total_chronics = len(self.env.chronics_handler.subpaths)
        
        for _ in range(num_tasks):
            task = {
                'chronics_id': np.random.randint(0, total_chronics),
                'weather_conditions': np.random.choice(['normal', 'storm', 'heat_wave']),
            }
            tasks.append(task)
        return tasks

    # -------- Gym Methods --------
    def step(self, action):
        """Execute one time step within the environment."""
        obs, reward, done, info = self.gym_env.step(action)
        
        # Add task-specific information to info
        info.update({
            'task': self.task,
            'grid_state': self.env.get_grid_state(),  # Need base env for grid state...?
            'chronic_id': self.current_chronic_id
        })
        
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        """Reset the environment to initial state."""
        # If we have a chronic ID set, ensure it's used
        if self.current_chronic_id is not None:
            kwargs['options'] = {"time serie id": self.current_chronic_id}
        obs = self.gym_env.reset(*args, **kwargs)
        return obs

    def render(self, mode='human'):
        """Render the environment."""
        return self.gym_env.render(mode)

    def close(self):
        """Clean up environment resources."""
        self.gym_env.close()
        self.env.close()


if __name__ == '__main__':
    # Test the environment
    env = Grid2OpDirectionEnv(
        env_name="l2rpn_case14_sandbox",
        action_type="topology"
    )
    
    # Test with a specific chronic ID
    test_task = {
        'chronics_id': 42,  # Use a specific chronic ID
        'weather_conditions': 'normal'
    }
    
    print("Testing with specific chronic ID...")
    env.set_task(test_task)
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Task: {test_task}")
    print(f"Chronic ID in info: {info['chronic_id']}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    
    # Test with random task
    print("\nTesting with random task...")
    random_task = env.sample_tasks(1)[0]
    env.set_task(random_task)
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Task: {random_task}")
    print(f"Chronic ID in info: {info['chronic_id']}")
    print(f"Reward: {reward}")
    print(f"Done: {done}") 