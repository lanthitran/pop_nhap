import os
import argparse

from common.imports import np, gym, th
import grid2op
from lightsim2grid import LightSimBackend
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Import make_env from RL2Grid
from env.utils import make_env
from grid2op.Parameters import Parameters

def train_ppo(env_id="bus14", difficulty=0, action_type="topology", total_timesteps=1_000_000, save_path="trained_model"):
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
            self.use_heuristic = False
            self.seed = 42
    
    args = Args()
    
    # Create environment using make_env from RL2Grid
    env_creator = make_env(args, 0, params=params)
    env = env_creator()
    
    # Convert to SB3 compatible environment
    env = DummyVecEnv([lambda: env])
    
    # Define callback for saving models during training
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=f"./checkpoints/{env_id}_{difficulty}",
        name_prefix="ppo_model"
    )
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=20000,
        batch_size=128,
        n_epochs=0,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        policy_kwargs={"net_arch": [512, 256, 256]}
    )
    
    # Train model
    print(f"Training PPO agent for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model, env

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent with Stable Baselines 3")
    parser.add_argument("--env_id", type=str, default="bus14", help="Environment ID")
    parser.add_argument("--difficulty", type=int, default=0, help="Difficulty level")
    parser.add_argument("--action_type", type=str, default="topology", choices=["topology", "redispatch"], help="Action type")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timesteps for training")
    parser.add_argument("--save_path", type=str, default="trained_models/ppo_grid2op", help="Path to save model")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and load model from save_path")
    
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
            save_path=args.save_path
        )
        env.close()

if __name__ == "__main__":
    main() 