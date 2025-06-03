import os
import numpy as np
import grid2op
from grid2op.gym_compat import GymEnv, DiscreteActSpace
from lightsim2grid import LightSimBackend
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt


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


def plot_substation_impacts(g2op_env, all_analyses: List[Dict]):
    """
    Create a bar chart showing how many actions impact each substation.
    
    Args:
        g2op_env: The Grid2Op environment
        all_analyses: List of action analysis results
    """
    # Count impacts per substation
    substation_impacts = np.zeros(g2op_env.n_sub)
    for analysis in all_analyses:
        for sub_id in analysis['impacted_subs']:
            substation_impacts[sub_id] += 1
    
    # Print total number of substation-impacting actions
    total_sub_impacts = int(substation_impacts.sum())
    print(f"Total number of substation-impacting actions: {total_sub_impacts}")
    
    # Create figure with dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    bars = ax.bar(range(g2op_env.n_sub), substation_impacts)
    
    # Customize the plot
    ax.set_xlabel('Substation ID', fontsize=12)
    ax.set_ylabel('Number of Impacting Actions', fontsize=12)
    ax.set_title('Number of Actions Impacting Each Substation', fontsize=14, pad=20)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Customize grid and ticks
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(g2op_env.n_sub))
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def plot_line_impacts(g2op_env, all_analyses: List[Dict]):
    """
    Create a bar chart showing how many actions impact each power line.

    Args:
        g2op_env: The Grid2Op environment
        all_analyses: List of action analysis results
    """
    # Count impacts per line
    line_impacts = np.zeros(g2op_env.n_line)
    for analysis in all_analyses:
        for line_id in analysis['impacted_lines']:
            line_impacts[line_id] += 1

    # Print total number of line-impacting actions (sum of all impacts)
    total_line_impacts = int(line_impacts.sum())
    print(f"Total number of line-impacting actions: {total_line_impacts}")

    # Print number of actions that impact at least one line
    num_actions_with_line_impact = sum(1 for analysis in all_analyses if len(analysis['impacted_lines']) > 0)
    print(f"Number of actions that impact at least one line: {num_actions_with_line_impact}")

    # Create figure with dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar chart
    bars = ax.bar(range(g2op_env.n_line), line_impacts)

    # Customize the plot
    ax.set_xlabel('Line ID', fontsize=12)
    ax.set_ylabel('Number of Impacting Actions', fontsize=12)
    ax.set_title('Number of Actions Impacting Each Power Line', fontsize=14, pad=20)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    # Customize grid and ticks
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(g2op_env.n_line))

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


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
    all_analyses = []
    for action_id in range(n_actions):
        action = gym_env.action_space.from_gym(action_id)
        analysis = analyze_action(action, action_id, g2op_env)
        print_action_analysis(analysis)
        all_analyses.append(analysis)
    
    # Print summary
    print_summary_statistics(g2op_env, n_actions)
    
    # Create visualization
    plot_substation_impacts(g2op_env, all_analyses)
    plot_line_impacts(g2op_env, all_analyses)


if __name__ == "__main__":
    main()

