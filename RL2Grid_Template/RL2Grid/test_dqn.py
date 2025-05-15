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
from alg.dqn.config import get_alg_args

from grid2op import make as g2op_make
from grid2op.Agent import BaseAgent
from grid2op.Runner import Runner
from grid2op.Parameters import Parameters
from tqdm import tqdm # Import tqdm for the progress bar


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

class DQNAgentWrapper(BaseAgent):
    """
    A wrapper for the trained DQN QNetwork to make it compatible with grid2op.Runner.
    """
    def __init__(self, q_network, g2op_env_action_space, 
                 gym_obs_converter, gym_act_converter, device):
        """
        Args:
            q_network: The trained PyTorch QNetwork model.
            g2op_env_action_space: The native Grid2Op action space from the evaluation environment.
            gym_obs_converter: The Gymnasium observation space wrapper (e.g., BoxGymObsSpace)
                               used to convert Grid2Op observations to NumPy arrays.
            gym_act_converter: The Gymnasium action space wrapper (e.g., DiscreteActSpace)
                               used to convert action indices to Grid2Op actions.
            device: The torch device (e.g., "cpu", "cuda").
        """
        super().__init__(g2op_env_action_space)
        self.q_network = q_network
        self.gym_obs_converter = gym_obs_converter
        self.gym_act_converter = gym_act_converter
        self.device = device
        self.q_network.eval() # Ensure model is in evaluation mode

    def act(self, observation, reward, done):
        gym_obs_array = self.gym_obs_converter.to_gym(observation)
        obs_tensor = th.tensor(gym_obs_array, dtype=th.float32).unsqueeze(0).to(self.device)
        with th.no_grad():
            action_idx = self.q_network.get_action(obs_tensor)
            if isinstance(action_idx, th.Tensor): # If get_action returns a tensor
                action_idx = action_idx.cpu().item() # Get Python number
        
        grid2op_action = self.gym_act_converter.from_gym(action_idx)
        return grid2op_action

def test_agent(qnet, envs, agent_config_args, max_steps, eval_seed):
    """Tests the DQN agent in the environment."""
    device = th.device("cuda" if th.cuda.is_available() and agent_config_args.cuda else "cpu")
    qnet.to(device)
    total_rewards = []
    episode_lengths = []

    num_episodes = 10  # You can change the number of test episodes
    print(f"\nRunning {num_episodes} evaluation episodes...")

    for episode in range(num_episodes):
        obs, _ = envs.reset(seed=eval_seed + episode) # Use the dedicated eval_seed
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
    parser.add_argument("--eval-env-id", type=str, default="bus14_val",
                        help="The ID of the environment to use for evaluation (e.g., bus14_test, bus14_val). If None, uses --env-id.")
    parser.add_argument("--action-type", type=str, default="topology", help="Type of action space (e.g., topology, redispatch)")
    parser.add_argument("--env-config-path", type=str, default="scenario.json", help="Path to the environment configuration file")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation environment")

    # Torch
    parser.add_argument("--cuda", type=str2bool, default=False, help="Enable CUDA by default.")
    parser.add_argument("--n-threads", type=int, default=8, help="Max number of torch threads.")
    
    # Runner specific
    parser.add_argument("--runner-output-dir", type=str, default="grid2op_runner_results", help="Directory to save Grid2Op Runner outputs.")
    parser.add_argument("--num-runner-episodes", type=int, default=5, help="Number of episodes for Grid2Op Runner to evaluate.")

    # Environment parameters (simplified for testing, copy relevant levels from main_*.py)
    # These are used if you want to apply specific Grid2Op Parameters during evaluation
    # For simplicity, we'll use args.difficulty to pick a level.
    # You might want to make this more sophisticated or load from a shared config.
    global ENV_PARAMS_FOR_TEST # Make it global to be accessible for param creation
    ENV_PARAMS_FOR_TEST = {
        0: {}, # Default, no specific overrides beyond MAX_LINE/SUB_CHANGED
        1: {"HARD_OVERFLOW_THRESHOLD": 999, "NB_TIMESTEP_OVERFLOW_ALLOWED": 9999999, "SOFT_OVERFLOW_THRESHOLD": 1.0, "NO_OVERFLOW_DISCONNECTION": True},
        2: {"HARD_OVERFLOW_THRESHOLD": 999, "NB_TIMESTEP_OVERFLOW_ALLOWED": 30, "SOFT_OVERFLOW_THRESHOLD": 1.0},
        3: {"HARD_OVERFLOW_THRESHOLD": 999, "NB_TIMESTEP_OVERFLOW_ALLOWED": 20, "SOFT_OVERFLOW_THRESHOLD": 1.0},
        4: {"HARD_OVERFLOW_THRESHOLD": 999, "NB_TIMESTEP_OVERFLOW_ALLOWED": 10, "SOFT_OVERFLOW_THRESHOLD": 1.0},
        5: {"HARD_OVERFLOW_THRESHOLD": 2.0, "NB_TIMESTEP_OVERFLOW_ALLOWED": 3, "SOFT_OVERFLOW_THRESHOLD": 1.0}
    }
    parser.add_argument("--norm-obs", type=str2bool, default=False, help="Normalize observation")
    parser.add_argument("--use-heuristic", type=str2bool, default=True, help="Use heuristic")
    
    cmd_line_args = parser.parse_args()
    
    # Load the algorithm-specific arguments
    alg_args = get_alg_args()
    # Combine command line args with algorithm defaults
    args = ap.Namespace(**vars(cmd_line_args), **vars(alg_args))

    eval_seed_from_cmd = args.seed # Capture seed specifically for evaluation

    # Determine the environment ID for evaluation
    eval_env_id_to_use = args.eval_env_id if args.eval_env_id else args.env_id
    print(f"Setting up evaluation environment: {eval_env_id_to_use}")

    # --- Prepare arguments and parameters for environment creation ---
    env_creation_args = ap.Namespace(**vars(args)) # Start with a copy of current args
    env_creation_args.env_id = eval_env_id_to_use
    # env_creation_args.seed will be eval_seed_from_cmd (set below)

    grid_eval_params = Parameters()
    grid_eval_params.MAX_LINE_STATUS_CHANGED = 1 # Default from main_*.py
    grid_eval_params.MAX_SUB_CHANGED = 1       # Default from main_*.py

    param_level_for_eval = args.difficulty
    if param_level_for_eval in ENV_PARAMS_FOR_TEST:
        params_dict = ENV_PARAMS_FOR_TEST[param_level_for_eval]
        if params_dict: # Apply only if there are params for this level
            print(f"Runner: Using Grid2Op parameter set level {param_level_for_eval} for eval environment:")
            for key, value in params_dict.items():
                print(f"  {key}: {value}")
                setattr(grid_eval_params, key, value)
    else:
        print(f"Runner: Warning: Invalid param level {param_level_for_eval} for Grid2Op Parameters. Using defaults + MAX_LINE/SUB_CHANGED=1.")

    # --- Create a Gym-wrapped environment to get converters and for QNetwork init ---
    # This GymEnv will use the eval_env_id and grid_eval_params
    # We use a single environment (idx=0) and the eval_seed
    env_creation_args.seed = eval_seed_from_cmd # Ensure make_env uses the eval seed
    temp_gym_env_thunk = make_env(env_creation_args, idx=0, resume_run=False, params=grid_eval_params)
    gym_env_for_eval_config = temp_gym_env_thunk()

    # This AsyncVectorEnv is primarily for QNetwork initialization if it expects a VectorEnv interface
    # It's based on the evaluation environment's configuration.
    vec_env_for_qnet_init = gym.vector.AsyncVectorEnv([
        lambda: make_env(env_creation_args, i, resume_run=False, params=grid_eval_params)() for i in range(1)
    ])

    set_random_seed(eval_seed_from_cmd) # Use eval_seed for global random state during testing
    set_torch(args.n_threads, True, args.cuda)
    device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

    # Load the checkpoint
    # QNetwork will be initialized using the observation/action spaces from vec_env_for_qnet_init
    qnet, loaded_checkpoint_args = load_checkpoint(args.checkpoint_path, vec_env_for_qnet_init, args)
    qnet.to(device)
    
    print(f"Successfully loaded checkpoint. Model was trained with env_id: {loaded_checkpoint_args.env_id} and seed: {loaded_checkpoint_args.seed}")
    print(f"Evaluating on env_id: {eval_env_id_to_use} with seed: {eval_seed_from_cmd}")

    # --- Setup for Grid2Op Runner ---
    g2op_eval_env = gym_env_for_eval_config.init_env # Get the raw Grid2Op environment

    # Instantiate the agent wrapper
    agent_wrapper = DQNAgentWrapper(
        q_network=qnet,
        g2op_env_action_space=g2op_eval_env.action_space,
        gym_obs_converter=gym_env_for_eval_config.observation_space, # BoxGymObsSpace
        gym_act_converter=gym_env_for_eval_config.action_space,   # DiscreteActSpace
        device=device
    )

    # Initialize Runner as per the requested pattern
    runner = Runner(**g2op_eval_env.get_params_for_runner(),
                    agentClass=None, # We provide an instance
                    agentInstance=agent_wrapper)
    


    print(f"\nStarting Grid2Op Runner evaluation for {args.num_runner_episodes} episodes...")
    
    # Call runner.run() with parameters as per the requested pattern
    results_summary = runner.run(nb_episode=args.num_runner_episodes,
                                 max_iter=-1,  # -1 means run episodes to their natural end, or use a specific MAX_EVAL_STEPS if defined
                                 pbar_tqdm_class=tqdm, # Pass the tqdm class for progress bar
                                 path_save=args.runner_output_dir)
 
    print("Grid2Op Runner evaluation finished.")
    print(f"Results summary: {results_summary}")
    print(f"Detailed logs and results saved in: {args.runner_output_dir}")

    # Clean up
    gym_env_for_eval_config.close()
    vec_env_for_qnet_init.close()
    # g2op_eval_env is owned by gym_env_for_eval_config, so it should be closed by its wrapper.





# python c:/Users/Admin/pop_nhap/RL2Grid_Template/RL2Grid/test_dqn.py --checkpoint-path "C:/path/to/your/model/checkpoint.pt"  --num-runner-episodes 30  --runner-output-dir "./my_dqn_evaluation_results"  --seed 123    --eval-env-id "bus14_eval"   --env-id "bus14_train"  --action-type "topology" --difficulty 0  --cuda True --env-config-path "scenario.json" --norm-obs True --use-heuristic True 



