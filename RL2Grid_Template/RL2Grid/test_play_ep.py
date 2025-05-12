import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend
import numpy as np
import matplotlib.pyplot as plt
from grid2op.Agent import DoNothingAgent
from grid2op.gym_compat import DiscreteActSpace
import os
import time

def evaluate_topology_action(env, action, max_steps=None, verbose=True):
    """
    Evaluate a single topology action by applying it once and then using do-nothing actions.
    Returns the normalized overflow factors for each line.
    """
    # Reset environment
    obs = env.reset()
    
    if max_steps is None:
        max_steps = env.chronics_handler.max_episode_duration()
    
    # Initialize counters and metrics
    step = 0
    done = False
    n_lines = env.n_line
    overflow_timesteps = np.zeros(n_lines, dtype=int)
    
    # Apply the topology action
    if verbose:
        print(f"Applying topology action...")
    
    obs, reward, done, info = env.step(action)
    step += 1
    
    # Track overflow for each line after first action
    for line_idx in range(n_lines):
        if obs.rho[line_idx] >= 1.0 and obs.line_status[line_idx]:
            overflow_timesteps[line_idx] += 1
    
    # For the rest of the episode: do nothing
    if verbose:
        print("Now using do-nothing actions for the remainder of the episode")
    
    do_nothing_action = env.action_space({})  # Empty dict = do nothing
    
    # Continue the episode with do-nothing actions
    while not done and step < max_steps:
        # Apply do-nothing action
        obs, reward, done, info = env.step(do_nothing_action)
        
        # Count lines in overflow
        for line_idx in range(n_lines):
            if obs.rho[line_idx] >= 1.0 and obs.line_status[line_idx]:
                overflow_timesteps[line_idx] += 1
        
        # Print status occasionally if verbose
        if verbose and step % 50 == 0:
            print(f"Step {step}/{max_steps}: max_rho={obs.rho.max():.2f}, "
                  f"active_lines={sum(obs.line_status)}/{env.n_line}")
            lines_in_overflow = sum(1 for i in range(n_lines) if obs.rho[i] >= 1.0 and obs.line_status[i])
            if lines_in_overflow > 0:
                print(f"  WARNING: {lines_in_overflow} lines currently in overflow!")
        
        step += 1
    
    # Calculate normalized overflow factors (sum = 1)
    total_overflow = np.sum(overflow_timesteps)
    if total_overflow > 0:
        overflow_factors = overflow_timesteps / total_overflow
    else:
        overflow_factors = np.zeros(n_lines)  # If no overflows, all factors are 0
    
    # If episode terminated early, print reason if verbose
    if done and verbose:
        print(f"Episode terminated early after {step} steps")
        if 'exception' in info:
            print(f"Reason: {info['exception'].__class__.__name__}")
    
    return {
        'overflow_timesteps': overflow_timesteps,
        'overflow_factors': overflow_factors,
        'total_steps': step,
        'survival_rate': step / max_steps * 100,
        'early_termination': done
    }

def main():
    # Start timing
    start_time = time.time()
    
    # Create parameters based on level 1 (easiest)
    params = Parameters()
    params.HARD_OVERFLOW_THRESHOLD = 999
    params.NB_TIMESTEP_OVERFLOW_ALLOWED = 9999999
    params.SOFT_OVERFLOW_THRESHOLD = 1.0
    params.NO_OVERFLOW_DISCONNECTION = True
    params.MAX_LINE_STATUS_CHANGED = 1
    params.MAX_SUB_CHANGED = 1
    
    print("Creating environment...")
    env = grid2op.make(
        "l2rpn_case14_sandbox",
        backend=LightSimBackend(),
        param=params
    )
    
    # Load the precalculated action space from numpy file
    action_space_path = os.path.join("env", "action_spaces/bus14_action_space.npy")
    
    print(f"Loading action space from: {action_space_path}")
    loaded_action_space = np.load(action_space_path, allow_pickle=True)
    print(f"Loaded {len(loaded_action_space)} predefined actions")
    
    # Create DiscreteActSpace
    discrete_action_space = DiscreteActSpace(
        env.action_space,
        action_list=loaded_action_space
    )
    
    # Get number of lines in the grid
    n_lines = env.n_line
    print(f"Grid has {n_lines} lines and {env.n_sub} substations")
    
    # Decide whether to evaluate a single action or all actions
    evaluate_all = input("Evaluate all topology actions? (y/n, default: n): ").lower().strip()
    evaluate_all = evaluate_all == 'y'
    
    if evaluate_all:
        # Set the number of actions to evaluate
        n_actions_to_evaluate = int(input(f"How many actions to evaluate? (1-{len(loaded_action_space)}, default: 20): ") or "20")
        n_actions_to_evaluate = min(n_actions_to_evaluate, len(loaded_action_space))
        print(f"Will evaluate {n_actions_to_evaluate} actions.")
        
        # Create array to store results
        # Shape: (n_actions, n_lines) - for each action, store overflow factors for each line
        all_overflow_factors = np.zeros((n_actions_to_evaluate, n_lines))
        all_results = []  # Store detailed results for each action
        
        # For speeding up evaluation, use shorter episodes
        max_steps = int(input("Max steps per episode (default: 500): ") or "500")
        
        # Evaluate each action
        for action_idx in range(n_actions_to_evaluate):
            print(f"\nEvaluating action {action_idx}/{n_actions_to_evaluate-1}...")
            action = discrete_action_space.from_gym(action_idx)
            
            # Evaluate with less verbose output
            result = evaluate_topology_action(env, action, max_steps=max_steps, verbose=False)
            
            # Store the overflow factors
            all_overflow_factors[action_idx] = result['overflow_factors']
            
            # Store full results
            all_results.append({
                'action_idx': action_idx,
                'overflow_timesteps': result['overflow_timesteps'],
                'overflow_factors': result['overflow_factors'],
                'total_steps': result['total_steps'],
                'survival_rate': result['survival_rate'],
                'early_termination': result['early_termination']
            })
            
            # Print brief summary
            total_overflow = sum(result['overflow_timesteps'])
            print(f"  Survived {result['survival_rate']:.1f}% of max episode length")
            print(f"  Total overflow timesteps: {total_overflow}")
            print(f"  Early termination: {result['early_termination']}")
            
            # Show progress
            if action_idx % 5 == 0 or action_idx == n_actions_to_evaluate - 1:
                elapsed = time.time() - start_time
                estimated_total = elapsed / (action_idx + 1) * n_actions_to_evaluate
                print(f"Progress: {action_idx+1}/{n_actions_to_evaluate} actions evaluated")
                print(f"Time elapsed: {elapsed:.1f}s, Estimated total: {estimated_total:.1f}s")
        
        # Save all results
        np.save('all_topology_overflow_factors.npy', all_overflow_factors)
        print("Saved overflow factors for all actions to 'all_topology_overflow_factors.npy'")
        
        # Analyze results to find best and worst actions
        total_overflows = np.sum(np.array([r['overflow_timesteps'] for r in all_results]), axis=1)
        survival_rates = np.array([r['survival_rate'] for r in all_results])
        
        # Find actions with no overflows and best survival rate
        no_overflow_mask = total_overflows == 0
        if np.any(no_overflow_mask):
            best_no_overflow_idx = np.argmax(survival_rates * no_overflow_mask)
            print(f"\nBest action with no overflows: {best_no_overflow_idx}")
            print(f"  Survival rate: {survival_rates[best_no_overflow_idx]:.1f}%")
        
        # Find action with highest survival rate
        best_survival_idx = np.argmax(survival_rates)
        print(f"\nAction with best survival rate: {best_survival_idx}")
        print(f"  Survival rate: {survival_rates[best_survival_idx]:.1f}%")
        print(f"  Total overflow timesteps: {total_overflows[best_survival_idx]}")
        
        # Save detailed analysis
        with open('topology_analysis_summary.txt', 'w') as f:
            f.write("Action\tSurvival\tTotal Overflow\tEarly Term\n")
            for idx, result in enumerate(all_results):
                f.write(f"{idx}\t{result['survival_rate']:.1f}%\t{sum(result['overflow_timesteps'])}\t{result['early_termination']}\n")
        print("Detailed analysis saved to 'topology_analysis_summary.txt'")
        
    else:
        # Evaluate just a single action (typically action 1, the first non-do-nothing action)
        first_action_idx = int(input("Enter action index to evaluate (default: 1): ") or "1")
        
        # Get the action
        action = discrete_action_space.from_gym(first_action_idx)
        action_dict = action
        print(f"Applying topology action (index {first_action_idx}):")
        print(f"Action details: {action_dict}")
        
        # Evaluate the action
        result = evaluate_topology_action(env, action)
        overflow_factors = result['overflow_factors']
        overflow_timesteps = result['overflow_timesteps']
        
        # Save the overflow factors to numpy file
        np.save(f'overflow_factors_action_{first_action_idx}.npy', overflow_factors)
        print(f"Saved overflow factors to 'overflow_factors_action_{first_action_idx}.npy'")
        
        # Print overflow statistics
        print("\n----- Overflow Statistics -----")
        print(f"Action {first_action_idx} - Line overflow statistics:")
        total_overflow = sum(overflow_timesteps)
        if total_overflow > 0:
            print(f"Total overflow-timesteps across all lines: {total_overflow}")
            print("\nOverflow timesteps by line:")
            for line_idx in range(n_lines):
                if overflow_timesteps[line_idx] > 0:
                    line_name = f"Line_{line_idx}"
                    percentage = (overflow_timesteps[line_idx] / result['total_steps']) * 100
                    factor = overflow_factors[line_idx]
                    print(f"  {line_name}: {overflow_timesteps[line_idx]} steps ({percentage:.1f}% of episode), factor: {factor:.4f}")
        else:
            print("No lines experienced overflow during the episode.")
        
        # Save overflow data to CSV
        overflow_data = np.column_stack((
            np.arange(n_lines),  # Line indices
            overflow_timesteps,  # Overflow timesteps
            (overflow_timesteps / result['total_steps']) * 100,  # Percentage of time in overflow
            overflow_factors  # Normalized factors (sum = 1)
        ))
        np.savetxt(
            f'line_overflow_stats_action_{first_action_idx}.csv', 
            overflow_data, 
            delimiter=',', 
            header='line_idx,overflow_timesteps,overflow_percentage,overflow_factor',
            comments=''
        )
        print(f"Overflow statistics saved to 'line_overflow_stats_action_{first_action_idx}.csv'")
        
        # Plot the overflow factors
        plt.figure(figsize=(12, 6))
        plt.bar(range(n_lines), overflow_factors)
        plt.xlabel('Line Index')
        plt.ylabel('Overflow Factor (normalized)')
        plt.title(f'Normalized Overflow Factors for Action {first_action_idx}')
        plt.tight_layout()
        plt.savefig(f'overflow_factors_action_{first_action_idx}.png')
        print(f"Overflow factors plot saved to 'overflow_factors_action_{first_action_idx}.png'")
    
    # Close environment
    env.close()
    
    # Print total execution time
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()
