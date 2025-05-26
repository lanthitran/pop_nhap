from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_policy_search.envs.normalized_env import normalize
from meta_policy_search.meta_algos.pro_mp import ProMP
from meta_policy_search.meta_trainer import Trainer
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder

import numpy as np
import tensorflow as tf
import os
import json
import argparse
import time

"""
This is the main script for running ProMP (Proximal Meta-Policy Search) on MuJoCo environments.
It sets up the training pipeline including environment, policy, sampler, and trainer components.

Key components:
1. Environment: MuJoCo-based environment with randomized parameters
2. Policy: Meta-learning policy that can adapt to new tasks
3. Sampler: Collects experience data for training
4. Trainer: Orchestrates the meta-learning process

The script handles both configuration loading and training execution.

Detailed Component Descriptions:
1. Environment: Uses HalfCheetah with randomized directions to create diverse tasks
2. Policy: Gaussian MLP policy that can adapt to new tasks through meta-learning
3. Sampler: Collects trajectories for multiple tasks in parallel
4. Trainer: Manages the meta-learning loop and optimization
5. ProMP Algorithm: Implements proximal meta-policy search with PPO-style updates

Key Features:
- Supports both file-based and default configurations
- Implements parallel sampling for efficiency
- Uses normalized environments for better training
- Includes comprehensive logging and checkpointing
| Hung |
"""

# Get the root path of the meta_policy_search project           | Hung |
meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def main(config):
    """
    Main training function that initializes and runs the ProMP algorithm
    
    This function sets up the complete training pipeline:
    1. Initializes baseline for advantage estimation
    2. Creates and normalizes the environment
    3. Builds the meta-learning policy
    4. Sets up sampling and processing components
    5. Configures the ProMP algorithm
    6. Creates and runs the trainer
    
    Args:
        config (dict): Configuration dictionary containing all hyperparameters
        and training settings
    | Hung |
    """
    # Set random seed for reproducibility         | Hung |
    set_seed(config['seed'])

    # Initialize baseline for advantage estimation         | Hung |
    baseline = globals()[config['baseline']]() # instantiate baseline

    # Create and normalize environment         | Hung |
    env = globals()[config['env']]() # instantiate env
    env = normalize(env) # apply normalize wrapper to env

    # Create meta-learning policy with specified architecture         | Hung |
    policy = MetaGaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
        )

    # Initialize sampler for collecting experience         | Hung |
    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )

    # Create sample processor for handling collected data         | Hung |
    sample_processor = MetaSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
    )

    # Initialize ProMP algorithm with specified parameters         | Hung |
    algo = ProMP(
        policy=policy,
        inner_lr=config['inner_lr'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        learning_rate=config['learning_rate'],
        num_ppo_steps=config['num_promp_steps'],
        clip_eps=config['clip_eps'],
        target_inner_step=config['target_inner_step'],
        init_inner_kl_penalty=config['init_inner_kl_penalty'],
        adaptive_inner_kl_penalty=config['adaptive_inner_kl_penalty'],
    )

    # Create trainer to orchestrate the meta-learning process         | Hung |
    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
    )

    # Start training         | Hung |
    trainer.train()

if __name__=="__main__":
    # Generate unique run ID based on timestamp         | Hung |
    idx = int(time.time())

    # Set up command line argument parsing         | Hung |
    parser = argparse.ArgumentParser(description='ProMP: Proximal Meta-Policy Search')
    parser.add_argument('--config_file', type=str, default='', 
                       help='json file with run specifications')
    parser.add_argument('--dump_path', type=str, 
                       default=meta_policy_search_path + '/data/pro-mp/run_%d' % idx)

    args = parser.parse_args()

    # Load configuration from file or use defaults         | Hung |
    if args.config_file: # load configuration from json file
        with open(args.config_file, 'r') as f:
            config = json.load(f)

    else: # use default config

        config = {
            'seed': 1,

            'baseline': 'LinearFeatureBaseline',

            'env': 'HalfCheetahRandDirecEnv',

            # sampler config
            'rollouts_per_meta_task': 20,  # Number of rollouts per meta-task         | Hung |
            'max_path_length': 100,        # Maximum trajectory length         | Hung |
            'parallel': True,              # Enable parallel sampling         | Hung |

            # sample processor config
            'discount': 0.99,              # Discount factor for future rewards         | Hung |
            'gae_lambda': 1,               # GAE lambda parameter         | Hung |
            'normalize_adv': True,         # Whether to normalize advantages         | Hung |

            # policy config
            'hidden_sizes': (64, 64),      # MLP architecture         | Hung |
            'learn_std': True, # whether to learn the standard deviation of the gaussian policy

            # ProMP config
            'inner_lr': 0.1, # adaptation step size
            'learning_rate': 1e-3, # meta-policy gradient step size
            'num_promp_steps': 5, # number of ProMp steps without re-sampling
            'clip_eps': 0.3, # clipping range
            'target_inner_step': 0.01,     # Target KL divergence for inner updates         | Hung |
            'init_inner_kl_penalty': 5e-4, # Initial KL penalty coefficient         | Hung |
            'adaptive_inner_kl_penalty': False, # whether to use an adaptive or fixed KL-penalty coefficient
            'n_itr': 1001, # number of overall training iterations
            'meta_batch_size': 40, # number of sampled meta-tasks per iterations
            'num_inner_grad_steps': 1, # number of inner / adaptation gradient steps

        }

    # configure logger
    logger.configure(dir=args.dump_path, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')

    # dump run configuration before starting training
    # Save configuration to file          | Hung |
    json.dump(config, open(args.dump_path + '/params.json', 'w'), cls=ClassEncoder)

    # start the actual algorithm
    main(config)