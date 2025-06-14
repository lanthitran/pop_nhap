#!/usr/bin/env python3

import gym
import numpy as np
from learn2learn.gym.envs.meta_env import MetaEnv
from .RL2Grid.env.utils import make_env, make_env_for_gym




class Grid2OpDirectionEnv(MetaEnv, gym.Env, gym.utils.EzPickle):
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

    def __init__(self, task=None, env_name="bus14", action_type="topology", **kwargs):
        # Create arguments for make_env
        class Args:
            def __init__(self):
                self.env_id = env_name
                self.action_type = action_type
                self.env_config_path = kwargs.get("env_config_path", "scenario.json")
                self.norm_obs = kwargs.get("norm_obs", True)
                self.use_heuristic = kwargs.get("use_heuristic", True)
                self.heuristic_type = kwargs.get("heuristic_type", "idle")
                self.seed = kwargs.get("seed", 42)
                self.difficulty = kwargs.get("difficulty", 0)
                self.reward_fn = kwargs.get("reward_fn", ["L2RPNRewardRegularized"])
                self.reward_factors = kwargs.get("reward_factors", [1.0])
                print(f"[Grid2OpDirectionEnv] Using reward_fn: {self.reward_fn}, reward_factors: {self.reward_factors}")
        
        args = Args()
        
        # Create environment using make_env (already returns a Gym env)
        env_creator = make_env_for_gym(args, 0)   # or make_env ?
        self.env = env_creator()  # This is already a Gym-compatible env
        
        # Set observation and action spaces directly from self.env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Store current chronic ID
        self.current_chronic_id = None
        
        MetaEnv.__init__(self, task)
        gym.Env.__init__(self)
        gym.utils.EzPickle.__init__(self)
        self.task = task
        #self.set_task(self.task)       #Don't need anymore, MetaEnv.__init__ already does it

    # -------- MetaEnv Methods --------
    def set_task(self, task):
        """Set the current task configuration."""
        if task is None:
            return
        MetaEnv.set_task(self, task)
        
        # Update environment parameters based on task
        if 'chronics_id' in task:
            # Store the chronic ID
            self.current_chronic_id = task['chronics_id']
            # Reset environment with the specified chronic ID
            self.env.reset(options={"time serie id": self.current_chronic_id})
            
        if 'weather_conditions' in task:
            # TODO 
            # Note: Weather conditions need to be set on the base Grid2Op environment
            # as it's not exposed through the Gym interface
            if hasattr(self.env.init_env, 'set_weather_conditions'):
                self.env.init_env.set_weather_conditions(task['weather_conditions'])  # not work yet
        
        # Update self.task after all modifications
        self.task = task

    def sample_tasks(self, num_tasks):
        """Sample different Grid2Op scenarios as tasks."""
        tasks = []
        # Access chronics_handler through the base Grid2Op environment
        chronic_names = self.env.init_env.chronics_handler.subpaths
        
        for _ in range(num_tasks):
            task = {
                'chronics_id': np.random.choice(chronic_names),  # Randomly select a chronic name
                'weather_conditions': np.random.choice(['normal', 'storm', 'heat_wave']),
            }
            tasks.append(task)
        return tasks

    # -------- Gym Methods --------
    def step(self, action):
        """Execute one time step within the environment."""
        obs, reward, done, info = self.env.step(action)
        
        # Add task-specific information to info
        info.update({
            'task': self.task,
            'grid_state': self.env.init_env.get_grid_state() if hasattr(self.env.init_env, 'get_grid_state') else None,
            'chronic_id': self.current_chronic_id
        })
        
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        """Reset the environment to initial state."""
        # If we have a chronic ID set, ensure it's used
        if self.current_chronic_id is not None:
            kwargs['options'] = {"time serie id": self.current_chronic_id}
        obs = self.env.reset(*args, **kwargs)
        return obs

    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render(mode)

    def close(self):
        """Clean up environment resources."""
        self.env.close()
""" 
if __name__ == '__main__':
    # Test the environment with custom arguments
    env = Grid2OpDirectionEnv(
        task=None,
        env_name="bus14",
        action_type="topology",
        env_config_path="scenario.json",
        norm_obs=True,
        use_heuristic=True,
        heuristic_type="idle",
        seed=42,
        difficulty=0,
        reward_fn=["L2RPNRewardRegularized"],   # L2RPNRewardRegularized
        reward_factors=[1.0]
    )
    
    # Deactivate forecast before any environment interaction
    if hasattr(env.env, 'init_env'):
        env.env.init_env.deactivate_forecast()
    else:
        print("Warning: Could not deactivate forecast - init_env not found")

    # Test with a specific chronic ID
    test_task = {
        'chronics_id': 35,  # Use a specific chronic ID
        'weather_conditions': 'normal'
    }

    print("Testing with specific chronic ID...")
    env.set_task(test_task)
    
    # Set random seed for action sampling
    np.random.seed(42)
    
    # Number of episodes to run
    num_episodes = 300
    
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action = 0  #env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            print(f"\nStep {step_count}:")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print(f"Total Return: {total_reward}")
            print(f"Done: {done}")
            
            if done:
                print(f"\nEpisode {episode + 1} finished after {step_count} steps")
                print(f"Episode length: {info['episode']['l'][0]}")
                print(f"Episode return: {info['episode']['r'][0]}")
                print(f"Info: {info}")

    env.close()


"""