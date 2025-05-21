from time import time


from alg.ppo.core import PPO
from alg.sac.core import SAC
from alg.dqn.core import DQN
from alg.td3.core import TD3
from common.checkpoint import PPOCheckpoint, SACCheckpoint, DQNCheckpoint, TD3Checkpoint
from common.utils import set_random_seed, set_torch, str2bool
from common.imports import ap, gym, th, np
from env.config import get_env_args
from env.utils import make_env_with_L2RPNReward, make_env_TOPOLOGY_with_L2RPNReward, make_env_TOPOLOGY_IDLE_BSLINE, make_env_TOPOLOGY_IDLE_RootReward
from grid2op.Parameters import Parameters

ALGORITHMS  = {'DQN': DQN, 'PPO': PPO, 'SAC': SAC, 'TD3': TD3}

# Define difficulty parameter presets
ENV_PARAMS = {
    # Easy parameters 
    1: {
        "HARD_OVERFLOW_THRESHOLD": 999,
        "NB_TIMESTEP_OVERFLOW_ALLOWED": 9999999,
        "SOFT_OVERFLOW_THRESHOLD": 1.0,
        "NO_OVERFLOW_DISCONNECTION": True
    },
    2: {
        "HARD_OVERFLOW_THRESHOLD": 999,
        "NB_TIMESTEP_OVERFLOW_ALLOWED": 70,
        "SOFT_OVERFLOW_THRESHOLD": 1.0
    },
    3: {
        "HARD_OVERFLOW_THRESHOLD": 999,
        "NB_TIMESTEP_OVERFLOW_ALLOWED": 20,
        "SOFT_OVERFLOW_THRESHOLD": 1.0
    },
    4: {
        "HARD_OVERFLOW_THRESHOLD": 999,
        "NB_TIMESTEP_OVERFLOW_ALLOWED": 10,
        "SOFT_OVERFLOW_THRESHOLD": 1.0
    },
    5: {
        "HARD_OVERFLOW_THRESHOLD": 2.0,
        "NB_TIMESTEP_OVERFLOW_ALLOWED": 3,
        "SOFT_OVERFLOW_THRESHOLD": 1.0
    }
}

# *** CHOOSE THE DIFFICULTY LEVEL HERE ***
# Set to a number between 1 (easiest) and 5 (hardest)
PARAM_LEVEL = 5

def main(args):
    assert args.time_limit <= 100440, f"Invalid time limit: {args.time_limit}. Timeout limit is : 100440"
    start_time = time()
    
    args = ap.Namespace(**vars(args), **vars(get_env_args()))
    assert args.n_envs >= 1, f"Invalid n° of environments: {args.n_envs}. Must be >= 1"
    
    alg = args.alg.upper()
    assert alg in ALGORITHMS.keys(), f"Unsupported algorithm: {alg}. Supported algorithms are: {ALGORITHMS}"
    
    run_name = args.resume_run_name if args.resume_run_name \
        else f"{args.alg}_{args.env_id}_{args.exp_tag}_{args.seed}_{args.difficulty}_{int(time())}_{np.random.randint(0, 50000)}"

    if alg == 'PPO': checkpoint = PPOCheckpoint(run_name, args)
    elif alg == 'SAC': checkpoint = SACCheckpoint(run_name, args)
    elif alg == 'DQN': checkpoint = DQNCheckpoint(run_name, args)
    elif alg == 'TD3': checkpoint = TD3Checkpoint(run_name, args)
    else:
        pass    # TODO

    if checkpoint.resumed: 
        args = checkpoint.loaded_run['args']
        # If additional timesteps are specified, add them to total_timesteps
        if args.additional_timesteps > 0:
            print(f"Adding {args.additional_timesteps} additional timesteps to training")
            args.total_timesteps += args.additional_timesteps * 1
    
    # Create and configure the Parameters object
    grid_params = Parameters()
    
    # Always set these parameters
    grid_params.MAX_LINE_STATUS_CHANGED = 1
    grid_params.MAX_SUB_CHANGED = 1
    
    # Set difficulty-specific parameters
    if PARAM_LEVEL in ENV_PARAMS:
        params_dict = ENV_PARAMS[PARAM_LEVEL]
        print(f"Using parameter set level {PARAM_LEVEL}:")
        for key, value in params_dict.items():
            print(f"  {key}: {value}")
            setattr(grid_params, key, value)
    else:
        print(f"Warning: Invalid param level {PARAM_LEVEL}. Using default parameters.")
    
    #envs = gym.vector.SyncVectorEnv([make_env(args, i, resume_run=checkpoint.resumed, params=grid_params) for i in range(args.n_envs)])
    envs = gym.vector.AsyncVectorEnv([make_env_TOPOLOGY_IDLE_RootReward(args, i, resume_run=checkpoint.resumed, params=grid_params) for i in range(args.n_envs)])
    dummy_env = envs.env_fns[0]()
    max_steps = dummy_env.init_env.chronics_handler.max_episode_duration()
    dummy_env.close()

    set_random_seed(args.seed)
    set_torch(args.n_threads, args.th_deterministic, args.cuda)

    ALGORITHMS[alg](envs, max_steps, run_name, start_time, args, checkpoint)

    

    


if __name__ == "__main__":
    parser = ap.ArgumentParser()

    # Cluster
    parser.add_argument("--time-limit", type=float, default=100300, help="Time limit for the action ranking")
    parser.add_argument("--checkpoint", type=str2bool, default=True, help="Toggles checkpoint.")
    parser.add_argument("--resume-run-name", type=str, default='', help="Run name to resume")
    parser.add_argument("--additional-timesteps", type=int, default=100000, help="Additional timesteps to add when resuming training")

    # Reproducibility
    parser.add_argument("--alg", type=str, default='DQN', help="Algorithm to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Logger
    parser.add_argument("--verbose", type=str2bool, default=True, help="Toggles prints")
    parser.add_argument("--exp-tag", type=str, default='', help="Tag for logging the experiment")
    parser.add_argument("--track", type=str2bool, default=True, help="Tag for logging the experiment")
    parser.add_argument("--wandb-project", type=str, default="rl2grid_bugfinder", help="Wandb's project name.")
    parser.add_argument("--wandb-entity", type=str, default="", help="Entity (team) of wandb's project.")
    parser.add_argument("--wandb-mode", type=str, default="offline", help="Online or offline wandb mode.")

    # Torch
    parser.add_argument("--th-deterministic", type=str2bool, default=False, help="Enable deterministic in Torch.")
    parser.add_argument("--cuda", type=str2bool, default=False, help="Enable CUDA by default.")
    parser.add_argument("--n-threads", type=int, default=8, help="Max number of torch threads.")

    main(parser.parse_known_args()[0])
