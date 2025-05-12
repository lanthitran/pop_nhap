import time
import numpy as np
import torch as th
import gymnasium as gym
from env.utils import make_env  # To create the environment
from alg.dqn.agent import QNetwork  # To load the QNetwork architecture
from common.checkpoint import DQNCheckpoint #to load the checkpoint
from common.utils import set_random_seed, set_torch, str2bool
from common.imports import ap
from env.config import get_env_args
from alg.dqn.config import get_alg_args #import here

def load_checkpoint(checkpoint_path, envs, args):
    """Loads the DQN checkpoint from the given path."""
    checkpoint = th.load(checkpoint_path)
    
    # Create a QNetwork instance with the same architecture as the trained one
    qnet = QNetwork(envs, args)

    # Load the state dictionary into the qnet
    qnet.load_state_dict(checkpoint['qnet'])

    # Set the qnet to evaluation mode
    qnet.eval()

    return qnet, checkpoint['args']

def test_agent(qnet, envs, args, max_steps):
    """Tests the DQN agent in the environment."""
    device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")
    qnet.to(device)
    total_rewards = []
    episode_lengths = []

    num_episodes = 10  # You can change the number of test episodes

    for episode in range(num_episodes):
        obs, _ = envs.reset(seed=args.seed+episode)
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            with th.no_grad():
                actions = qnet.get_action(th.tensor(obs).to(device))
            
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            done = np.any(terminations) or np.any(truncations)
            total_reward += np.sum(rewards)
            episode_length += 1
            obs = next_obs
            if episode_length >= max_steps:
                done = True
        
        total_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Length = {episode_length}")

    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    print(f"\nAverage Total Reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(f"Average Episode Length over {num_episodes} episodes: {avg_length:.2f}")

if __name__ == "__main__":
    parser = ap.ArgumentParser()

    # Checkpoint
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the checkpoint file")

    # Environment
    parser.add_argument("--env-id", type=str, default="bus14", help="Environment ID (e.g., bus14, bus36-M)")
    parser.add_argument("--difficulty", type=int, default=0, help="Difficulty level of the environment")
    parser.add_argument("--action-type", type=str, default="topology", help="Type of action space (e.g., topology, redispatch)")
    parser.add_argument("--env-config-path", type=str, default="scenario.json", help="Path to the environment configuration file")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Torch
    parser.add_argument("--cuda", type=str2bool, default=False, help="Enable CUDA by default.")
    parser.add_argument("--n-threads", type=int, default=8, help="Max number of torch threads.")
    parser.add_argument("--norm-obs", type=str2bool, default=False, help="Normalize observation")
    parser.add_argument("--use-heuristic", type=str2bool, default=False, help="Use heuristic")
    
    args = parser.parse_args()
    #args = ap.Namespace(**vars(args), **vars(get_env_args())) #remove this line
    
    # Load the algorithm-specific arguments
    alg_args = get_alg_args()
    args = ap.Namespace(**vars(args), **vars(alg_args)) #add this line

    # Create the environment
    envs = gym.vector.AsyncVectorEnv([make_env(args, i, resume_run=False) for i in range(1)])
    dummy_env = envs.env_fns[0]()
    max_steps = dummy_env.init_env.chronics_handler.max_episode_duration()
    dummy_env.close()

    set_random_seed(args.seed)
    set_torch(args.n_threads, True, args.cuda)

    # Load the checkpoint
    qnet, loaded_args = load_checkpoint(args.checkpoint_path, envs, args)
    #args = ap.Namespace(**vars(args), **vars(loaded_args)) #remove this line
    args = loaded_args #add this line

    # Test the agent
    test_agent(qnet, envs, args, max_steps)
    envs.close()
