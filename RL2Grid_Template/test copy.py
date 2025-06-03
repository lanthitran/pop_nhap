import os
import numpy as np
import grid2op
from grid2op.gym_compat import GymEnv, DiscreteActSpace
from lightsim2grid import LightSimBackend
from typing import Tuple, List, Dict, Any


def create_environment(env_id: str, difficulty: int) -> Tuple[Any, GymEnv]:
    """
    Create and configure the Grid2Op and Gym environments.
    
    Args:
        env_id: The environment ID (e.g., 'bus_14')
        difficulty: The difficulty level (0-2)
        
    Returns:
        Tuple containing the Grid2Op environment and Gym environment
    """
    # Create the base environment
    g2op_env = grid2op.make(
        "l2rpn_case14_sandbox",
        #reward_class=None,
        backend=LightSimBackend()
    )
    
    # Create gym environment
    gym_env = GymEnv(g2op_env)
        
    # Load the action space
    loaded_action_space = np.load(f"D:/grid2op/pop_nhap/RL2Grid_Template/RL2Grid/env/action_spaces/{env_id}_action_space.npy", allow_pickle=True)
    #loaded_action_space = np.load(f"D:\grid2op\pop_nhap\RL2Grid_Template\RL2Grid/env/action_spaces/{env_id}_action_space.npy", allow_pickle=True)

    # Calculate number of actions based on difficulty
    n_actions = np.geomspace(50, len(loaded_action_space), num=3).astype(int)[difficulty]
    
    # Set up the action space
    gym_env.action_space = DiscreteActSpace(
        g2op_env.action_space,
        action_list=loaded_action_space[:n_actions]
    )
    
    return g2op_env, gym_env


def analyze_action(action, action_id: int, g2op_env) -> Dict:
    """
    Analyze a single action and return its impact details.
    
    Args:
        action: The action to analyze
        action_id: The ID of the action
        g2op_env: The Grid2Op environment
        
    Returns:
        Dictionary containing analysis results
    """
    # Get topological impact using the proper API method
    lines_impacted, subs_impacted = action.get_topological_impact()
    
    # Get impacted elements
    impacted_subs = np.where(subs_impacted)[0]
    impacted_lines = np.where(lines_impacted)[0]
    
    return {
        'action_id': action_id,
        'description': str(action),
        'impacted_subs': impacted_subs,
        'impacted_lines': impacted_lines
    }


def print_action_analysis(analysis: Dict):
    """
    Print the analysis results for a single action.
    
    Args:
        analysis: Dictionary containing action analysis results
    """
    print(f"\nAction {analysis['action_id']}:")
    print("-"*40)
    print(f"Action description: {analysis['description']}")
    
    if len(analysis['impacted_subs']) > 0:
        print("\nImpacted substations:")
        for sub_id in analysis['impacted_subs']:
            print(f"  - Substation {sub_id}")
    
    if len(analysis['impacted_lines']) > 0:
        print("\nImpacted power lines:")
        for line_id in analysis['impacted_lines']:
            print(f"  - Line {line_id}")
    
    print("-"*40)


def print_summary_statistics(g2op_env, n_actions: int):
    """
    Print summary statistics about the environment and action space.
    
    Args:
        g2op_env: The Grid2Op environment
        n_actions: Number of actions in the space
    """
    print("\nSummary Statistics:")
    print("="*80)
    print(f"Total number of actions: {n_actions}")
    print(f"Number of substations: {g2op_env.n_sub}")
    print(f"Number of power lines: {g2op_env.n_line}")


def main():
    """Main function to run the action space analysis."""
    # Configuration
    env_id = "bus14"
    difficulty = 0
    
    # Create environments
    g2op_env, gym_env = create_environment(env_id, difficulty)
    
    # Get number of actions
    n_actions = gym_env.action_space.n
    
    # Print header
    print(f"\nAnalyzing {n_actions} actions for {env_id} at difficulty {difficulty}\n")
    print("="*80)
    
    # Analyze each action
    for action_id in range(n_actions):
        action = gym_env.action_space.from_gym(action_id)
        analysis = analyze_action(action, action_id, g2op_env)
        print_action_analysis(analysis)
    
    # Print summary
    print_summary_statistics(g2op_env, n_actions)


if __name__ == "__main__":
    main()

