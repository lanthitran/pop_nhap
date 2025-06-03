




# i want to load the action space of grid2op, bus_14, difficulty 0, and print each action in the action space
# and then i want a statistical analysis of the action space, i want to know
# that for each action, what object does it impact (what substation)
# just print everything in a readable format
# how to create and load action space is in the env/utils.py file, pls do it like that
# but simpler, i only care about the action space of bus_14, difficulty 0
# how to know its impact 
'''
an action is an object of class BaseAction
in the docs, it says:
__str__()→ str[source]
This utility allows printing in a human-readable format what objects will be impacted by the action.

Returns
:
str – The string representation of an BaseAction in a human-readable format.

Return type
:
str

Examples

It is simply the “print” function:

action = env.action_space(...)
print(action)
_subs_impacted
This attributes is either not initialized (set to None) or it tells, for each substation, if it is impacted by the action (in this case BaseAction._subs_impacted[sub_id] is True) or not (in this case BaseAction._subs_impacted[sub_id] is False)

Type
:
numpy.ndarray, dtype:bool

_lines_impacted
This attributes is either not initialized (set to None) or it tells, for each powerline, if it is impacted by the action (in this case BaseAction._lines_impacted[line_id] is True) or not (in this case BaseAction._subs_impacted[line_id] is False)

Type
:
numpy.ndarray, dtype:bool
'''


import os
import sys
import json
import numpy as np
import grid2op
from grid2op.gym_compat import GymEnv, DiscreteActSpace
from lightsim2grid import LightSimBackend
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt


def load_scenario_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def create_environment(env_id: str, difficulty: int, scenario_config: dict, action_space_dir: str) -> Tuple[Any, GymEnv, dict]:
    env_info = scenario_config['environments'][env_id]
    grid2op_id = env_info['grid2op_id']
    max_difficulty = env_info['difficulty']

    g2op_env = grid2op.make(
        grid2op_id,
        backend=LightSimBackend(),
        test=True
    )
    gym_env = GymEnv(g2op_env)

    loaded_action_space = np.load(
        f"{action_space_dir}/{env_id}_action_space.npy", allow_pickle=True
    )
    n_actions = np.geomspace(50, len(loaded_action_space), num=max_difficulty).astype(int)[difficulty]

    gym_env.action_space = DiscreteActSpace(
        g2op_env.action_space,
        action_list=loaded_action_space[:n_actions]
    )

    return g2op_env, gym_env, {
        'env_id': env_id,
        'grid2op_id': grid2op_id,
        'max_difficulty': max_difficulty,
        'difficulty': difficulty,
        'n_actions': n_actions,
        'n_sub': g2op_env.n_sub,
        'n_line': g2op_env.n_line
    }


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


def print_summary_statistics(env_stats: dict):
    print("\nSummary Statistics:")
    print("="*80)
    print(f"Environment: {env_stats['env_id']} (grid2op_id: {env_stats['grid2op_id']})")
    print(f"Difficulty: {env_stats['difficulty']} / {env_stats['max_difficulty']-1}")
    print(f"Number of actions: {env_stats['n_actions']}")
    print(f"Number of substations: {env_stats['n_sub']}")
    print(f"Number of power lines: {env_stats['n_line']}")


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
    
    # Print number of actions that impact at least one substation
    num_actions_with_sub_impact = sum(1 for analysis in all_analyses if len(analysis['impacted_subs']) > 0)
    print(f"Number of actions that impact at least one substation: {num_actions_with_sub_impact}")
    
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
    # Config
    scenario_config_path = "D:/grid2op/pop_nhap/RL2Grid_Template/RL2Grid/env/scenario.json"
    action_space_dir = "D:/grid2op/pop_nhap/RL2Grid_Template/RL2Grid/env/action_spaces"

    # Available env_id: bus14, bus14_train, bus14_test, bus14_val, bus36-M, bus36-MO-v0, bus36-MO-v1, bus118-M, bus118-MOB-v0, bus118-MOB-v1
    env_id = "bus14"
    difficulty = 0

    scenario_config = load_scenario_config(scenario_config_path)
    g2op_env, gym_env, env_stats = create_environment(env_id, difficulty, scenario_config, action_space_dir)

    print(f"\nAnalyzing {env_stats['n_actions']} actions for {env_stats['env_id']} at difficulty {env_stats['difficulty']}\n")
    print("="*80)

    # Analyze each action
    all_analyses = []
    for action_id in range(env_stats['n_actions']):
        action = gym_env.action_space.from_gym(action_id)
        analysis = analyze_action(action, action_id, g2op_env)
        print_action_analysis(analysis)
        all_analyses.append(analysis)

    # Print summary
    print_summary_statistics(env_stats)

    # Create visualizations
    plot_substation_impacts(g2op_env, all_analyses)
    plot_line_impacts(g2op_env, all_analyses)


if __name__ == "__main__":
    main()

