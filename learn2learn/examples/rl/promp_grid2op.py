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
from learn2learn.gym.envs import Grid2OpDirectionEnv
from policies import DiagNormalPolicy


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
        env_name='bus14',  # Grid2Op environment
        action_type='topology',           # Action type (topology/redispatch)
        adapt_lr=0.1,                     # Learning rate for adaptation
        meta_lr=3e-4,                     # Learning rate for meta-updates
        adapt_steps=3,                    # Number of adaptation steps
        num_iterations=10,              # Total training iterations
        meta_bsz=40,                      # Meta-batch size
        adapt_bsz=20,                     # Adaptation batch size
        ppo_clip=0.3,                     # PPO clipping parameter
        ppo_steps=5,                      # PPO update steps
        tau=1.00,                         # GAE parameter
        gamma=0.99,                       # Discount factor
        eta=0.0005,                       # KL penalty coefficient
        adaptive_penalty=False,           # Whether to use adaptive KL penalty
        kl_target=0.01,                   # Target KL divergence
        num_workers=4,                    # Number of parallel workers
        seed=42,                          # Random seed
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
        """
        Creates and configures the Grid2Op environment.
        """
        import grid2op
        _ = grid2op.make('l2rpn_case14_sandbox', test=True)
        env = Grid2OpDirectionEnv(
            env_name=env_name,
            action_type=action_type
        )
        #env = ch.envs.ActionSpaceScaler(env)  # is this necessary? this will create a bug with .shape, there would be no shape

        return env

    # Initialize parallel environments
    #env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])  # TODO I can't resolve the pickling problem
    env = make_env()


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
    policy = DiagNormalPolicy(
        input_size=env.state_size,
        output_size=env.action_size,
        hiddens=[256, 256],  # Larger network for Grid2Op
        activation='tanh'
    )
    meta_learner = l2l.algorithms.MAML(policy, lr=meta_lr)
    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(meta_learner.parameters(), lr=meta_lr)

    # Main training loop
    for iteration in range(num_iterations):
        print(f"\n========== Iteration {iteration} ==========")
        iteration_reward = 0.0
        iteration_replays = []
        iteration_policies = []

        # Sample Trajectories
        # Sample trajectories for meta-batch
        print(">>> Sampling tasks and collecting trajectories (meta-batch)...")
        for task_config in tqdm(env.sample_tasks(meta_bsz), leave=False, desc='Data'):
            clone = deepcopy(meta_learner)
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)
            task_replay = []
            task_policies = []

            # Fast Adapt
            print("=== Adaptation Phase ===")
            # Fast adaptation loop with tqdm
            for step in tqdm(range(adapt_steps), leave=False, desc='Adapt', position=2):
                # Prepare policy for adaptation
                print(f"  [Adapt Step {step+1}/{adapt_steps}]")

                for p in clone.parameters():
                    p.detach_().requires_grad_()
                task_policies.append(deepcopy(clone))
                
                # Run parallel environments to collect adapt_bsz total episodes
                train_episodes = task.run(clone, episodes=adapt_bsz)
                clone = fast_adapt_a2c(clone, train_episodes, adapt_lr,
                                     baseline, gamma, tau, first_order=True)
                task_replay.append(train_episodes)

            # Compute Validation Loss
            # Compute validation performance
            print("=== Validation Phase ===")

            for p in clone.parameters():
                p.detach_().requires_grad_()
            task_policies.append(deepcopy(clone))
            valid_episodes = task.run(clone, episodes=adapt_bsz)
            task_replay.append(valid_episodes)
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz
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