






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
import numpy as np
import grid2op
from grid2op.gym_compat import GymEnv, DiscreteActSpace
from lightsim2grid import LightSimBackend

# Load the environment
env_id = "bus_14"
difficulty = 0
ENV_DIR = os.path.dirname(__file__)

# Create the environment
g2op_env = grid2op.make(
    "l2rpn_case14_sandbox",  # This is the grid2op_id for bus_14
    reward_class=None,  # We don't need rewards for this analysis
    backend=LightSimBackend()
)

# Create gym environment
gym_env = GymEnv(g2op_env)

# Load the action space
loaded_action_space = np.load(f"{ENV_DIR}/action_spaces/{env_id}_action_space.npy", allow_pickle=True)

# Calculate number of actions based on difficulty
n_actions = np.geomspace(50, len(loaded_action_space), num=3).astype(int)[difficulty]  # 3 is max_difficulty for bus_14

# Set up the action space
gym_env.action_space = DiscreteActSpace(
    g2op_env.action_space,
    action_list=loaded_action_space[:n_actions]
)

# Print analysis of each action
print(f"\nAnalyzing {n_actions} actions for {env_id} at difficulty {difficulty}\n")
print("="*80)

for action_id in range(n_actions):
    action = gym_env.action_space.from_gym(action_id)
    
    # Get impacted objects
    action._subs_impacted = np.zeros(g2op_env.n_sub, dtype=bool)
    action._lines_impacted = np.zeros(g2op_env.n_line, dtype=bool)
    action._update_impacted_elements()
    
    # Print action details
    print(f"\nAction {action_id}:")
    print("-"*40)
    print(f"Action description: {action}")
    
    # Print impacted substations
    impacted_subs = np.where(action._subs_impacted)[0]
    if len(impacted_subs) > 0:
        print("\nImpacted substations:")
        for sub_id in impacted_subs:
            print(f"  - Substation {sub_id}")
    
    # Print impacted lines
    impacted_lines = np.where(action._lines_impacted)[0]
    if len(impacted_lines) > 0:
        print("\nImpacted power lines:")
        for line_id in impacted_lines:
            print(f"  - Line {line_id}")
    
    print("-"*40)

# Print summary statistics
print("\nSummary Statistics:")
print("="*80)
print(f"Total number of actions: {n_actions}")
print(f"Number of substations: {g2op_env.n_sub}")
print(f"Number of power lines: {g2op_env.n_line}")