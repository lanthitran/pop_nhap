import os
import argparse
import numpy as np

from common.imports import np, gym, th

import grid2op
from lightsim2grid import LightSimBackend
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Import make_env from RL2Grid
from env.utils import make_env
from grid2op.Parameters import Parameters

def train_ppo(env_id="bus14", difficulty=0, action_type="topology", total_timesteps=1_000_000, save_path="trained_model", n_envs=1):
    """Train a PPO agent using Stable Baselines 3"""
    
    # Create parameters with appropriate difficulty level
    params = Parameters()
    params.MAX_LINE_STATUS_CHANGED = 1
    params.MAX_SUB_CHANGED = 1
    
    # Configure arguments for make_env
    class Args:
        def __init__(self):
            self.env_id = env_id
            self.difficulty = difficulty
            self.action_type = action_type
            self.env_config_path = "scenario.json"
            self.norm_obs = True
            self.use_heuristic = True #in the past i used to forgot to use this, which made the 
            self.seed = 42
    
    args = Args()
    
    # Create environment list
    env_fns = [make_env(args, i, params=params) for i in range(n_envs)]
    
    if n_envs > 1:
        # Use AsyncVectorEnv for parallel environments (matching main.py approach)
        #envs = gym.vector.AsyncVectorEnv(env_fns)
        print(f"Using {n_envs} parallel environments")
        #example: vec_env = DummyVecEnv([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
        envs = DummyVecEnv(env_fns)
    else:
        # For a single environment, we'll still use DummyVecEnv from stable_baselines3
        env_creator = make_env(args, 0, params=params)
        env = env_creator()
        envs = DummyVecEnv([lambda: env])
    
    # Print some info about the environment
    dummy_env = env_fns[0]()
    print(f"\nEnvironment spaces:")
    print(f"Observation space: {dummy_env.observation_space}")    
    print(f"Observation shape: {dummy_env.observation_space.shape}")
    print(f"Action space: {dummy_env.action_space}")
    if hasattr(dummy_env.action_space, "n"):
        print(f"Action space size: {dummy_env.action_space.n}")
    dummy_env.close()
    env_simple = make_env(args, 0, params=params)
    # Define callback for saving models during training
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=f"./checkpoints_STB_BSL/{env_id}_{difficulty}",
        name_prefix="ppo_model"
    )
    
    # Keep n_steps constant regardless of environment count
    n_steps = 20000  # From config.py, no longer scaling with n_envs
    
    # Create PPO model with parameters matching config.py
    model = PPO(
        policy="MlpPolicy",
        env=envs, #envs  env_simple
        learning_rate=2.5e-4,            # From config.py: actor_lr and critic_lr
        n_steps=20000,                 # From config.py: 20000 (constant regardless of env count)
        batch_size=128,  # (n_steps * n_envs) // 1563
        n_epochs=40,                     # From config.py: update_epochs
        gamma=0.9,                       # From config.py: gamma
        gae_lambda=0.95,                 # From config.py: gae_lambda
        clip_range=0.2,                  # From config.py: clip_coef
        normalize_advantage=True,        # From config.py: norm_adv
        ent_coef=0.01,                   # From config.py: entropy_coef
        vf_coef=0.5,                     # From config.py: vf_coef
        max_grad_norm=10,                # From config.py: max_grad_norm
        verbose=True,                  # Verbose level (0, 1, or 2)
        tensorboard_log="./ppo_STB_BL_tensorboard/",  # TensorBoard log directory
        policy_kwargs={
            "net_arch": {
                "pi": [256, 256],   # From config.py: actor_layers
                "vf": [256, 256]    # From config.py: critic_layers
            }
        }
    )
    
    # Train model
    print(f"Training PPO agent for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model, envs

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent with Stable Baselines 3")
    parser.add_argument("--env_id", type=str, default="bus14_train", help="Environment ID")
    parser.add_argument("--difficulty", type=int, default=0, help="Difficulty level")
    parser.add_argument("--action_type", type=str, default="topology", choices=["topology", "redispatch"], help="Action type")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timesteps for training")
    parser.add_argument("--save_path", type=str, default="trained_STB_BSL_models/ppo_grid2op", help="Path to save model")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and load model from save_path")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    
    args = parser.parse_args()
    
    if args.skip_training:
        # Load model
        print(f"Loading model from {args.save_path}")
        model = PPO.load(args.save_path)
        print("Model loaded successfully")
    else:
        # Train model
        model, env = train_ppo(
            env_id=args.env_id,
            difficulty=args.difficulty,
            action_type=args.action_type,
            total_timesteps=args.total_timesteps,
            save_path=args.save_path,
            n_envs=args.n_envs
        )
        env.close()

if __name__ == "__main__":
    main() 