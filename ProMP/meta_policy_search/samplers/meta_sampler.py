"""
This module implements the MetaSampler class which is responsible for collecting experience data 
from multiple environments in parallel for meta-reinforcement learning.

Key components:
1. MetaSampler: Main class that handles sampling trajectories from multiple tasks
2. Vectorized environment execution: Supports both parallel and iterative execution
3. Path collection: Manages the collection and organization of trajectories

The sampler is crucial for meta-RL as it enables efficient data collection across multiple tasks,
which is essential for learning transferable policies.
| Hung |
"""

from meta_policy_search.samplers.base import Sampler
from meta_policy_search.samplers.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from meta_policy_search.utils import utils, logger
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools


class MetaSampler(Sampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_policy_search.envs.base.MetaEnv) : environment object
        policy (meta_policy_search.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of envs to run vectorized for each task (influences the memory usage)
    """
    """
        Sampler for Meta-RL that handles collecting trajectories from multiple tasks simultaneously.
    
    This class is responsible for:
    1. Managing parallel environment execution
    2. Collecting trajectories from multiple tasks
    3. Organizing experience data for meta-learning
    
    The sampler supports both parallel and sequential execution modes for flexibility
    in different computational environments.
    | Hung |
    """
    
    def __init__(
            self,
            env,
            policy,
            rollouts_per_meta_task,
            meta_batch_size,
            max_path_length,
            envs_per_task=None,
            parallel=False
            ):
        # Initialize base sampler with environment and policy          | Hung |
        super(MetaSampler, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert hasattr(env, 'set_task')  # Ensure environment supports task setting          | Hung |

        # Configure sampling parameters          | Hung |
        self.envs_per_task = rollouts_per_meta_task if envs_per_task is None else envs_per_task
        self.meta_batch_size = meta_batch_size
        self.total_samples = meta_batch_size * rollouts_per_meta_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0

        # setup vectorized environment
        # Initialize vectorized environment based on parallel flag          | Hung |
        if self.parallel:
            self.vec_env = MetaParallelEnvExecutor(env, self.meta_batch_size, 
                                                  self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MetaIterativeEnvExecutor(env, self.meta_batch_size, 
                                                   self.envs_per_task, self.max_path_length)

    def update_tasks(self):
        """
        Samples a new goal for each meta task
        """
        """
        Samples new tasks for each meta-task in the batch.
        
        This method is called at the start of each meta-iteration to generate
        new tasks for the agent to learn from.
        | Hung |
        """
        tasks = self.env.sample_tasks(self.meta_batch_size)
        assert len(tasks) == self.meta_batch_size
        self.vec_env.set_tasks(tasks)

    def obtain_samples(self, log=False, log_prefix=''):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns: 
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """
        """

        Collects trajectories from all tasks in the meta-batch.
        
        This is the main sampling loop that:
        1. Executes the policy in each environment
        2. Collects observations, actions, and rewards
        3. Organizes trajectories into paths
        4. Handles environment resets and episode termination
        | Hung |
        """

        # initial setup / preparation
        # Initialize data structures for collecting trajectories          | Hung |
        paths = OrderedDict()
        for i in range(self.meta_batch_size):
            paths[i] = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy

        # initial reset of envs
        # Reset all environments to start new episodes          | Hung |
        obses = self.vec_env.reset()
        
        while n_samples < self.total_samples:
            
            # execute policy
            # Get actions from policy for current observations          | Hung |
            t = time.time()
            obs_per_task = np.split(np.asarray(obses), self.meta_batch_size)
            actions, agent_infos = policy.get_actions(obs_per_task)
            policy_time += time.time() - t

            # step environments
            # Execute actions in environments          | Hung |
            t = time.time()
            actions = np.concatenate(actions) # stack meta batch        # Combine actions from all tasks  | Hung |
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            # Process and organize information dictionaries          | Hung |
            agent_infos, env_infos = self._handle_info_dicts(agent_infos, env_infos)

            new_samples = 0
            # Process each environment's step results          | Hung |
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                # append new samples to running paths
                # Record step data in running path          | Hung |
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                # if running path is done, add it to paths and empty the running path
                # If episode is done, save trajectory and reset          | Hung |
                if done:
                    paths[idx // self.envs_per_task].append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        env_infos=utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        # Update total samples collected and log timing if requested          | Hung |
        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        return paths

    def _handle_info_dicts(self, agent_infos, env_infos):
        """
        Processes and validates information dictionaries from policy and environment.
        
        Ensures proper formatting of agent and environment information for
        trajectory collection and logging.
        | Hung |
        """
        # if no env_infos, create empty dicts for each env          | Hung |
        if not env_infos:
            env_infos = [dict() for _ in range(self.vec_env.num_envs)]
        # if no agent_infos, create empty dicts for each env          | Hung |
        if not agent_infos:
            agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
        else:
            # if agent_infos are provided, stack them          | Hung |
            assert len(agent_infos) == self.meta_batch_size
            assert len(agent_infos[0]) == self.envs_per_task
            agent_infos = sum(agent_infos, [])  # stack agent_infos    # Combine agent infos from all tasks   | Hung |

        assert len(agent_infos) == self.meta_batch_size * self.envs_per_task == len(env_infos)
        return agent_infos, env_infos


def _get_empty_running_paths_dict():
    """
    Creates an empty dictionary structure for storing trajectory data.
    
    This structure is used to accumulate observations, actions, rewards,
    and other information during trajectory collection.
    | Hung |
    """
    return dict(observations=[], actions=[], rewards=[], env_infos=[], agent_infos=[])
