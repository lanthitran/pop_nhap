#!/usr/bin/env python3

"""
Trains a policy network with ProMP for Grid2Op environment.

Usage:
python examples/rl/promp_grid2op.py
"""

import random
from copy import deepcopy

import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c, ppo, trpo
from cherry.models.robotics import LinearValue
from torch import optim
from torch.distributions.kl import kl_divergence
from tqdm import tqdm

import learn2learn as l2l
from learn2learn.gym.envs.grid2op.grid2op_direction import Grid2OpDirectionEnv


#fix print_episode_lengths_from_replay, follow the api of MyOwnRunner to extract info with RecordEpisodeStatistics


def my_own_flatten_episodes(replay, episodes, num_workers):
    """
    Like cherry.wrappers.runner_wrapper.flatten_episodes, but robustly splits info fields
    that are lists/tuples of dicts (or other per-worker data) so that only the correct
    per-worker value is passed to ExperienceReplay.append for each worker.
    """
    import cherry as ch
    from collections.abc import Iterable
    from cherry._utils import _min_size, _istensorable

    flat_replay = ch.ExperienceReplay()
    worker_replays = [ch.ExperienceReplay() for w in range(num_workers)]
    flat_episodes = 0

    for sars in replay:
        # Reshape tensors to handle batched data
        state = sars.state.view(_min_size(sars.state))
        action = sars.action.view(_min_size(sars.action))
        reward = sars.reward.view(_min_size(sars.reward))
        next_state = sars.next_state.view(_min_size(sars.next_state))
        done = sars.done.view(_min_size(sars.done))

        # Get additional info fields
        fields = set(sars._fields) - {'state', 'action', 'reward', 'next_state', 'done'}

        # Process each worker's data
        for worker in range(num_workers):
            worker_specific_infos = {'runner_id': worker}
            for f_name in fields:
                value = getattr(sars, f_name)
                # If value is a list/tuple of length num_workers, take the per-worker value
                if isinstance(value, (list, tuple)) and len(value) == num_workers:
                    worker_specific_infos[f_name] = value[worker]
                elif _istensorable(value):
                    tvalue = ch.totensor(value)
                    tvalue_view = tvalue.view(_min_size(tvalue))
                    if tvalue_view.dim() > 0 and tvalue_view.size(0) == num_workers:
                        worker_specific_infos[f_name] = tvalue_view[worker]
                    else:
                        worker_specific_infos[f_name] = value
                else:
                    worker_specific_infos[f_name] = value

            worker_replays[worker].append(
                state[worker],
                action[worker],
                reward[worker],
                next_state[worker],
                done[worker],
                **worker_specific_infos,
            )
            # If episode is done, merge worker's replay into flat replay
            if bool(done[worker]):
                flat_replay += worker_replays[worker]
                worker_replays[worker] = ch.ExperienceReplay()
                flat_episodes += 1

            # Check if we've collected enough episodes
            if flat_episodes >= episodes:
                break

        if flat_episodes >= episodes:
            break

    return flat_replay


class MyOwnRunner(ch.envs.Runner):
    """
    Custom Runner that prints episode length and return when an episode ends.
    Handles both classic and Gymnasium-style vectorized envs (e.g., AsyncVectorEnv).
    Correctly extracts episode statistics from infos['final_info'] for vectorized envs.
    Stores the full info dict from env.step() under 'env_step_info' in ExperienceReplay.
    If episode statistics are available, stores them under 'env_episode_stats'.
    """
    def run(self, get_action, steps=None, episodes=None, render=False):
        if steps is None:
            steps = float('inf')
            if self.is_vectorized:
                self._needs_reset = True
        elif episodes is None:
            episodes = float('inf')
        else:
            msg = 'Either steps or episodes should be set.'
            raise Exception(msg)

        replay = ch.ExperienceReplay(vectorized=self.is_vectorized)
        collected_episodes = 0
        collected_steps = 0
        while True:
            if collected_steps >= steps or collected_episodes >= episodes:
                if self.is_vectorized and collected_episodes >= episodes:
                    replay = my_own_flatten_episodes(replay, episodes, self.num_envs)
                    self._needs_reset = True
                return replay
            if self._needs_reset:
                self.reset()
            info = {}
            action = get_action(self._current_state)
            if isinstance(action, tuple):
                skip_unpack = False
                if self.is_vectorized:
                    if len(action) > 2:
                        skip_unpack = True
                    elif len(action) == 2 and \
                            self.env.num_envs == 2 and \
                            not isinstance(action[1], dict):
                        action = (action, )
                if not skip_unpack:
                    if len(action) == 2:
                        info = action[1]
                        action = action[0]
                    elif len(action) == 1:
                        action = action[0]
                    else:
                        msg = 'get_action should return 1 or 2 values.'
                        raise NotImplementedError(msg)
            old_state = self._current_state
            state, reward, done, step_info = self.env.step(action)
            # --- Print episode length(s) and return(s) if done ---
            if self.is_vectorized and isinstance(step_info, dict) and "final_info" in step_info:
                # Gymnasium-style vectorized envs: step_info['final_info'] is a list of dicts (or None)
                final_infos = step_info.get("final_info", None)
                final_mask = step_info.get("_final_info", None)
                for idx, d in enumerate(done):
                    # Only print if this worker just finished an episode (final_info is present and has 'episode')
                    ep_info = None
                    if final_infos is not None and isinstance(final_infos, (list, tuple)) and idx < len(final_infos):
                        ep_info = final_infos[idx]
                    if ep_info and isinstance(ep_info, dict) and "episode" in ep_info:
                        ep = ep_info["episode"]
                        msg = f"Worker {idx} finished episode"
                        if "l" in ep and "r" in ep:
                            msg += f" of length {ep['l']} with return {ep['r']}"
                        elif "l" in ep:
                            msg += f" of length {ep['l']}"
                        elif "r" in ep:
                            msg += f" with return {ep['r']}"
                        print(msg)
            elif self.is_vectorized:
                # Fallback for older vectorized envs: try to treat step_info as a list of dicts
                for idx, d in enumerate(done):
                    if d:
                        ep_info = None
                        if isinstance(step_info, (list, tuple)) and idx < len(step_info):
                            ep_info = step_info[idx]
                        if ep_info and isinstance(ep_info, dict) and "episode" in ep_info:
                            ep = ep_info["episode"]
                            msg = f"Worker {idx} finished episode"
                            if "l" in ep and "r" in ep:
                                msg += f" of length {ep['l']} with return {ep['r']}"
                            elif "l" in ep:
                                msg += f" of length {ep['l']}"
                            elif "r" in ep:
                                msg += f" with return {ep['r']}"
                            print(msg)
            else:
                # Non-vectorized env: step_info is a dict
                if done:
                    if isinstance(step_info, dict) and "episode" in step_info:
                        ep = step_info["episode"]
                        msg = "Episode finished"
                        if "l" in ep and "r" in ep:
                            msg += f" with length {ep['l']} and return {ep['r']}"
                        elif "l" in ep:
                            msg += f" with length {ep['l']}"
                        elif "r" in ep:
                            msg += f" with return {ep['r']}"
                        print(msg)
            # --- Store step_info and episode stats in info dict for ExperienceReplay ---
            info = dict(info)  # ensure we don't mutate the original

            # Only store env_step_info and env_episode_stats if they are not dicts or lists of dicts
            # to avoid cherry.totensor trying to tensorize them.
            # Instead, store them under a key that ExperienceReplay.append will ignore for tensorization,
            # or filter them out before passing to append.

            # We'll filter out non-tensorable info fields before passing to replay.append
            safe_info = {}
            for k, v in info.items():
                # Only keep if not a dict and not a list/tuple of dicts
                if isinstance(v, dict):
                    continue
                if isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, dict) for x in v):
                    continue
                safe_info[k] = v

            # Store env_step_info and env_episode_stats as special attributes (not in **info)
            # so they are attached to the transition but not tensorized
            extra_attrs = {}
            extra_attrs['env_step_info'] = step_info
            # For vectorized envs, extract episode stats from 'final_info'
            if self.is_vectorized and isinstance(step_info, dict) and "final_info" in step_info:
                final_infos = step_info.get("final_info", None)
                env_episode_stats = []
                if final_infos is not None and isinstance(final_infos, (list, tuple)):
                    for ep_info in final_infos:
                        if isinstance(ep_info, dict) and "episode" in ep_info:
                            env_episode_stats.append(ep_info["episode"])
                        else:
                            env_episode_stats.append(None)
                extra_attrs['env_episode_stats'] = env_episode_stats
            elif self.is_vectorized:
                # Fallback for older vectorized envs: try to treat step_info as a list of dicts
                env_episode_stats = []
                if isinstance(step_info, (list, tuple)):
                    for ep_info in step_info:
                        if isinstance(ep_info, dict) and "episode" in ep_info:
                            env_episode_stats.append(ep_info["episode"])
                        else:
                            env_episode_stats.append(None)
                extra_attrs['env_episode_stats'] = env_episode_stats
            else:
                if isinstance(step_info, dict) and "episode" in step_info:
                    extra_attrs['env_episode_stats'] = step_info["episode"]

            if not self.is_vectorized and done:
                collected_episodes += 1
                self._needs_reset = True
            elif self.is_vectorized:
                collected_episodes += sum(done)

            # Append transition, passing only tensorable info fields as **safe_info,
            # and attach extra_attrs as attributes after appending.
            sars = replay.append(old_state, action, reward, state, done, **safe_info)
            # Attach extra attributes to the last transition in the replay
            # (ExperienceReplay.append does not return the transition, so we access it directly)
            if len(replay) > 0:
                for k, v in extra_attrs.items():
                    setattr(replay[-1], k, v)

            self._current_state = state
            if render:
                self.env.render()
            collected_steps += 1
            
def print_episode_lengths_from_replay(replay, prefix=""):
    """
    Print episode lengths and returns from a Cherry ExperienceReplay or list of transitions.
    Follows the extraction logic of MyOwnRunner: looks for 'env_episode_stats' or 'env_step_info'
    attributes, as set by RecordEpisodeStatistics, and prints episode length and return for each
    transition where done==True. Also prints the action taken at the end of the episode.
    """
    def action_to_str(action):
        # Try to convert action to a readable string
        import torch
        if isinstance(action, torch.Tensor):
            return action.detach().cpu().numpy().tolist()
        elif isinstance(action, (list, tuple)):
            return [action_to_str(a) for a in action]
        else:
            return str(action)

    # Lists to store episode statistics
    episode_lengths = []
    episode_returns = []

    for idx, sars in enumerate(replay):
        # Get done value (tensor or bool)
        done_val = getattr(sars, 'done', None)
        if done_val is not None:
            done_val = done_val.item() if hasattr(done_val, 'item') else done_val
        else:
            continue

        if not bool(done_val):
            continue

        # Get action for printing
        action_val = getattr(sars, 'action', None)
        action_str = f" | Action: {action_to_str(action_val)}" if action_val is not None else ""

        # Try to extract episode stats in the same way as MyOwnRunner
        # 1. Check for 'env_episode_stats' attribute (set by MyOwnRunner)
        episode_stats = getattr(sars, 'env_episode_stats', None)
        if episode_stats is not None:
            # For vectorized envs, this is a list (one per env), for non-vectorized, a dict
            if isinstance(episode_stats, list):
                for env_idx, ep in enumerate(episode_stats):
                    if isinstance(ep, dict):
                        msg = f"{prefix}Worker {env_idx} episode ended"
                        if "l" in ep and "r" in ep:
                            msg += f". Length: {ep['l'][0]}, Return: {ep['r'][0]:.2f}"
                            episode_lengths.append(ep['l'][0])
                            episode_returns.append(ep['r'][0])
                        elif "l" in ep:
                            msg += f". Length: {ep['l'][0]}"
                            episode_lengths.append(ep['l'][0])
                        elif "r" in ep:
                            msg += f". Return: {ep['r'][0]:.2f}"
                            episode_returns.append(ep['r'][0])
                        msg += action_str
                        print(msg)
            elif isinstance(episode_stats, dict):
                msg = f"{prefix}Episode ended"
                if "l" in episode_stats and "r" in episode_stats:
                    msg += f". Length: {episode_stats['l'][0]}, Return: {episode_stats['r'][0]:.2f}"
                    episode_lengths.append(episode_stats['l'][0])
                    episode_returns.append(episode_stats['r'][0])
                elif "l" in episode_stats:
                    msg += f". Length: {episode_stats['l'][0]}"
                    episode_lengths.append(episode_stats['l'][0])
                elif "r" in episode_stats:
                    msg += f". Return: {episode_stats['r'][0]:.2f}"
                    episode_returns.append(episode_stats['r'][0])
                msg += action_str
                print(msg)
            continue  # Prefer env_episode_stats if present

        # 2. Fallback: try to extract from env_step_info (for older/other runners)
        step_info = getattr(sars, 'env_step_info', None)
        if step_info is not None:
            # For vectorized envs, step_info may be dict with 'final_info' or list of dicts
            if isinstance(step_info, dict) and "final_info" in step_info:
                final_infos = step_info.get("final_info", None)
                if final_infos is not None and isinstance(final_infos, (list, tuple)):
                    for env_idx, ep_info in enumerate(final_infos):
                        if isinstance(ep_info, dict) and "episode" in ep_info:
                            ep = ep_info["episode"]
                            msg = f"{prefix}Worker {env_idx} episode ended"
                            if "l" in ep and "r" in ep:
                                msg += f". Length: {ep['l'][0]}, Return: {ep['r'][0]:.2f}"
                                episode_lengths.append(ep['l'][0])
                                episode_returns.append(ep['r'][0])
                            elif "l" in ep:
                                msg += f". Length: {ep['l'][0]}"
                                episode_lengths.append(ep['l'][0])
                            elif "r" in ep:
                                msg += f". Return: {ep['r'][0]:.2f}"
                                episode_returns.append(ep['r'][0])
                            msg += action_str
                            print(msg)
            elif isinstance(step_info, (list, tuple)):
                for env_idx, ep_info in enumerate(step_info):
                    if isinstance(ep_info, dict) and "episode" in ep_info:
                        ep = ep_info["episode"]
                        msg = f"{prefix}Worker {env_idx} episode ended"
                        if "l" in ep and "r" in ep:
                            msg += f". Length: {ep['l'][0]}, Return: {ep['r'][0]:.2f}"
                            episode_lengths.append(ep['l'][0])
                            episode_returns.append(ep['r'][0])
                        elif "l" in ep:
                            msg += f". Length: {ep['l'][0]}"
                            episode_lengths.append(ep['l'][0])
                        elif "r" in ep:
                            msg += f". Return: {ep['r'][0]:.2f}"
                            episode_returns.append(ep['r'][0])
                        msg += action_str
                        print(msg)
            elif isinstance(step_info, dict) and "episode" in step_info:
                ep = step_info["episode"]
                msg = f"{prefix}Episode ended"
                if "l" in ep and "r" in ep:
                    msg += f". Length: {ep['l'][0]}, Return: {ep['r'][0]:.2f}"
                    episode_lengths.append(ep['l'][0])
                    episode_returns.append(ep['r'][0])
                elif "l" in ep:
                    msg += f". Length: {ep['l'][0]}"
                    episode_lengths.append(ep['l'][0])
                elif "r" in ep:
                    msg += f". Return: {ep['r'][0]:.2f}"
                    episode_returns.append(ep['r'][0])
                msg += action_str
                print(msg)
            continue

        # 3. Fallback: try to extract from info dict (legacy Cherry API)
        info = getattr(sars, 'info', None)
        if isinstance(info, dict) and 'episode' in info:
            ep = info['episode']
            msg = f"{prefix}Episode ended"
            if "l" in ep and "r" in ep:
                msg += f". Length: {ep['l'][0]}, Return: {ep['r'][0]:.2f}"
                episode_lengths.append(ep['l'][0])
                episode_returns.append(ep['r'][0])
            elif "l" in ep:
                msg += f". Length: {ep['l'][0]}"
                episode_lengths.append(ep['l'][0])
            elif "r" in ep:
                msg += f". Return: {ep['r'][0]:.2f}"
                episode_returns.append(ep['r'][0])
            msg += action_str
            print(msg)
            continue

        # 4. Fallback: try to extract from direct attribute (legacy)
        episode_info = getattr(sars, 'episode', None)
        if isinstance(episode_info, dict):
            msg = f"{prefix}Episode ended"
            if "l" in episode_info and "r" in episode_info:
                msg += f". Length: {episode_info['l'][0]}, Return: {episode_info['r'][0]:.2f}"
                episode_lengths.append(episode_info['l'][0])
                episode_returns.append(episode_info['r'][0])
            elif "l" in episode_info:
                msg += f". Length: {episode_info['l'][0]}"
                episode_lengths.append(episode_info['l'][0])
            elif "r" in episode_info:
                msg += f". Return: {episode_info['r'][0]:.2f}"
                episode_returns.append(episode_info['r'][0])
            msg += action_str
            print(msg)
    # Print summary statistics using numpy for simpler mean calculation
    if episode_lengths:
        mean_length = np.mean(episode_lengths)
        print(f"\n{prefix} =========== Mean episode length: {mean_length:.2f}")
    if episode_returns:
        mean_return = np.mean(episode_returns)
        print(f"{prefix} =========== Mean episode return: {mean_return:.2f}")


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    """
    Computes the generalized advantage estimation (GAE) for policy optimization.
    Combines value function baseline with temporal difference learning.
    """
    # Update baseline with discounted returns
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    # Handle terminal states properly
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    # Compute advantages using GAE
    return ch.pg.generalized_advantage(tau=tau,
                                     gamma=gamma,
                                     rewards=rewards,
                                     dones=dones,
                                     values=bootstraps,
                                     next_value=next_value)


def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau):
    """
    Computes the A2C loss for MAML adaptation.
    Combines policy gradient with value function baseline.
    """
    # Extract episode data
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()
    
    # Compute policy loss with advantages
    log_probs = learner.log_prob(states, actions)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                  dones, states, next_states)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt_a2c(clone, train_episodes, adapt_lr, baseline, gamma, tau, first_order=False):
    """
    Performs fast adaptation using A2C algorithm.
    Updates policy parameters to maximize expected returns.
    """
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    clone.adapt(loss, first_order=first_order)
    return clone


def precompute_quantities(states, actions, old_policy, new_policy):
    """
    Precomputes policy distributions and log probabilities for PPO update.
    Used to compute KL divergence and policy loss efficiently.
    """
    old_density = old_policy.density(states)
    old_log_probs = old_density.log_prob(actions).mean(dim=1, keepdim=True).detach()
    new_density = new_policy.density(states)
    new_log_probs = new_density.log_prob(actions).mean(dim=1, keepdim=True)
    return old_density, new_density, old_log_probs, new_log_probs


def main(
        rl_env_name='Grid2OpDirection-v1',  # RL environment name
        env_name='bus14',                   # Grid2Op environment
        action_type='topology',             # Action type (topology/redispatch)
        adapt_lr=0.1,                       # Learning rate for adaptation
        meta_lr=3e-4,                       # Learning rate for meta-updates
        adapt_steps=5,                      # Number of adaptation steps
        num_iterations=10000,                  # Total training iterations
        meta_bsz=40,                        # Meta-batch size
        adapt_bsz=40,                       # Adaptation batch size
        ppo_clip=0.3,                       # PPO clipping parameter
        ppo_steps=5,                        # PPO update steps
        tau=1.00,                           # GAE parameter
        gamma=0.9,                         # Discount factor
        eta=0.0005,                         # KL penalty coefficient
        adaptive_penalty=False,             # Whether to use adaptive KL penalty
        kl_target=0.01,                     # Target KL divergence
        num_workers=10,                      # Number of parallel workers
        seed=42,                            # Random seed
):
    """
    Main training loop for ProMP algorithm with Grid2Op.
    Implements meta-learning with PPO constraints for stable adaptation.
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        # Each worker gets a different seed by using the worker index
        rd_num = random.randint(1, 99999999999)
        worker_seed = seed + rd_num
        env = gym.make(rl_env_name, env_name=env_name, action_type=action_type, seed=worker_seed)
        #env = ch.envs.ActionSpaceScaler(env)  # is this necessary? If execute this will create a bug with .shape, there would be no shape...
        return env
    

    # Initialize parallel environments
    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])  # TO_DO I can't resolve the pickling problem


    # Patch: Convert observation space if it's Gymnasium-based and Cherry expects Gym-based
    # This addresses the AssertionError in cherry.envs.utils.get_space_dimension
    if 'gymnasium' in str(type(env.observation_space)).lower(): # More robust check
        try:
            from gym.spaces import Box as LegacyGymBox # Cherry uses gym.spaces
            gymnasium_obs_space = env.observation_space
            env.observation_space = LegacyGymBox(
                low=gymnasium_obs_space.low,
                high=gymnasium_obs_space.high,
                shape=gymnasium_obs_space.shape,
                dtype=gymnasium_obs_space.dtype
            )
            print(f"Successfully patched env.observation_space to: {type(env.observation_space)}")
        except ImportError:
            print("Warning: gym.spaces.Box not available. Cannot patch observation_space for Cherry compatibility.")
        except Exception as e:
            print(f"Warning: Failed to patch env.observation_space: {e}")

    # Patch: Convert action space if it's Gymnasium-based and Cherry expects Gym-based
    if 'gymnasium' in str(type(env.action_space)).lower(): # More robust check
        try:
            from gym.spaces import Discrete as LegacyGymDiscrete
            gymnasium_act_space = env.action_space
            # Assuming the Gymnasium-based discrete space has an 'n' attribute
            if hasattr(gymnasium_act_space, 'n'):
                env.action_space = LegacyGymDiscrete(n=gymnasium_act_space.n)
                print(f"Successfully patched env.action_space to: {type(env.action_space)}")
            else:
                print(f"Warning: Original action space {type(gymnasium_act_space)} does not have 'n' attribute. Cannot patch.")  # TODO continous act space?
        except ImportError:
            print("Warning: gym.spaces.Discrete not available. Cannot patch action_space for Cherry compatibility.")
        except Exception as e:
            print(f"Warning: Failed to patch env.action_space: {e}")


    print(env)
    print("--------------------------------")
    env.seed(seed)
    env = ch.envs.Torch(env)
    # Initialize policy and meta-learner
    from policies import CategoricalPolicy, DiagNormalPolicy, CategoricalPolicyLegacy

    if action_type == 'topology':
        policy = CategoricalPolicy(
            input_size=env.state_size,
            output_size=env.action_size,
            hiddens=[256, 256]  # Larger network for Grid2Op
        )
    elif action_type == 'redispatch':
        policy = DiagNormalPolicy(
            input_size=env.state_size,
            output_size=env.action_size,
            hiddens=[256, 256],  # Larger network for Grid2Op
            device='cpu'  # or 'cuda' if using GPU
        )
    else:
        raise ValueError(f"Unknown action_type: {action_type}")
    
    meta_learner = l2l.algorithms.MAML(policy, lr=meta_lr)
    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(meta_learner.parameters(), lr=meta_lr)

    # Main training loop
    for iteration in range(num_iterations):
        print(f"\n============================== Iteration {iteration} ==============================")
        iteration_reward = 0.0
        iteration_replays = []
        iteration_policies = []

        # Sample Trajectories
        # Sample trajectories for meta-batch
        print(">>> Sampling tasks and collecting trajectories (meta-batch)...")
        for task_config in tqdm(env.sample_tasks(meta_bsz), leave=False, desc='Task_Data'):
            clone = deepcopy(meta_learner)
            env.set_task(task_config)
            env.reset()
            #task = ch.envs.Runner(env)
            task = MyOwnRunner(env)
            task_replay = []
            task_policies = []

            # Fast Adapt
            print("======================= Adaptation Phase =======================")
            # Fast adaptation loop with tqdm
            for step in tqdm(range(adapt_steps), leave=False, desc='Adapt', position=2):
                # Prepare policy for adaptation
                print(f" ====================== [Adapt Step {step+1}/{adapt_steps}] ======================")

                for p in clone.parameters():
                    p.detach_().requires_grad_()
                task_policies.append(deepcopy(clone))
                
                # Run parallel environments to collect adapt_bsz total episodes
                train_episodes = task.run(clone, episodes=adapt_bsz)

                # Print episode lengths using the compact method
                print_episode_lengths_from_replay(train_episodes)

                clone = fast_adapt_a2c(clone, train_episodes, adapt_lr,
                                     baseline, gamma, tau, first_order=True)
                task_replay.append(train_episodes)

            # Compute Validation Loss
            # Compute validation performance
            print("======================= Validation Phase =======================")

            for p in clone.parameters():
                p.detach_().requires_grad_()
            task_policies.append(deepcopy(clone))
            valid_episodes = task.run(clone, episodes=adapt_bsz)

            # Print episode lengths for validation episodes using the compact method
            print_episode_lengths_from_replay(valid_episodes, prefix="Validation ")

            task_replay.append(valid_episodes)
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz
            print(iteration_reward)
            iteration_replays.append(task_replay)
            iteration_policies.append(task_policies)

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_bsz
        print('adaptation_reward', adaptation_reward)

        # ProMP meta-optimization
        # ProMP meta-optimization with PPO
        print(">>> ProMP Meta-Optimization Phase (PPO steps)...")
        for ppo_iter in tqdm(range(ppo_steps), leave=False, desc='Optim'):
            print(f"  [Meta-Optimization Step {ppo_iter+1}/{ppo_steps}]")
            promp_loss = 0.0
            kl_total = 0.0
            for task_replays, old_policies in zip(iteration_replays, iteration_policies):
                new_policy = meta_learner.clone()
                states = task_replays[0].state()
                actions = task_replays[0].action()
                rewards = task_replays[0].reward()
                dones = task_replays[0].done()
                next_states = task_replays[0].next_state()
                old_policy = old_policies[0]
                
                # Precompute policy distributions
                (old_density,
                 new_density,
                 old_log_probs,
                 new_log_probs) = precompute_quantities(states,
                                                      actions,
                                                      old_policy,
                                                      new_policy)
                advantages = compute_advantages(baseline, tau, gamma, rewards,
                                             dones, states, next_states)
                advantages = ch.normalize(advantages).detach()
                
                # Adaptation steps with tqdm
                for step in tqdm(range(adapt_steps), leave=False, desc='ProMP Adapt', position=3):
                    print(f"    [ProMP Adapt Step {step+1}/{adapt_steps}]")
                    # Compute KL penalty
                    kl_pen = kl_divergence(old_density, new_density).mean()
                    kl_total += kl_pen.item()

                    # Update policy with TRPO
                    surr_loss = trpo.policy_loss(new_log_probs, old_log_probs, advantages)
                    new_policy.adapt(surr_loss)

                    # Move to next adaptation step
                    states = task_replays[step + 1].state()
                    actions = task_replays[step + 1].action()
                    rewards = task_replays[step + 1].reward()
                    dones = task_replays[step + 1].done()
                    next_states = task_replays[step + 1].next_state()
                    old_policy = old_policies[step + 1]
                    (old_density,
                     new_density,
                     old_log_probs,
                     new_log_probs) = precompute_quantities(states,
                                                          actions,
                                                          old_policy,
                                                          new_policy)

                    # Compute clip loss
                    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                                 dones, states, next_states)
                    advantages = ch.normalize(advantages).detach()
                    clip_loss = ppo.policy_loss(new_log_probs,
                                              old_log_probs,
                                              advantages,
                                              clip=ppo_clip)

                    # Combine into ProMP loss
                    promp_loss += clip_loss + eta * kl_pen

            kl_total /= meta_bsz * adapt_steps
            promp_loss /= meta_bsz * adapt_steps
            opt.zero_grad()
            promp_loss.backward(retain_graph=True)
            opt.step()

            # Adapt KL penalty based on desired target
            if adaptive_penalty:
                if kl_total < kl_target / 1.5:
                    eta /= 2.0
                elif kl_total > kl_target * 1.5:
                    eta *= 2.0


if __name__ == '__main__':
    main() 