import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from grid2op.Agent import DoNothingAgent
from collections import defaultdict

def find_overflow_intervals(overflow_array):
    """Find continuous intervals of overflows in a binary array"""
    intervals = []
    current_start = None
    
    for i, val in enumerate(overflow_array):
        if val == 1 and current_start is None:
            current_start = i
        elif val == 0 and current_start is not None:
            intervals.append((current_start, i-1))
            current_start = None
    
    # If episode ends during an overflow
    if current_start is not None:
        intervals.append((current_start, len(overflow_array)-1))
    
    return intervals

def play_episodes_and_analyze(nb_episodes=2):
    # Create parameters with NO_OVERFLOW_DISCONNECTION = True
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    
    # Create environment
    env = grid2op.make(
        "l2rpn_case14_sandbox",
        backend=LightSimBackend(),
        param=params
    )
    
    # Create do-nothing agent
    agent = DoNothingAgent(env.action_space)
    
    # Dictionary to store duration frequencies for each line across all episodes
    all_line_durations = defaultdict(list)
    
    for episode in range(nb_episodes):
        print(f"======= Episode {episode+1} ================")
        
        # Reset environment
        obs = env.reset()
        
        # Track line overflows
        n_lines = env.n_line
        line_names = [f"Line_{i}" for i in range(n_lines)]
        overflow_history = {name: [] for name in line_names}
        
        # Play episode
        done = False
        step = 0
        
        while not done:
            # Get action from agent (do-nothing)
            action = agent.act(obs, None, None)
            
            # Apply action
            obs, reward, done, info = env.step(action)
            
            # Record line overflows
            for line_idx in range(n_lines):
                is_overflow = obs.rho[line_idx] >= 1.0 and obs.line_status[line_idx]
                overflow_history[line_names[line_idx]].append(1 if is_overflow else 0)
            
            step += 1
        
        # Find overflow intervals for each line
        line_intervals = {}
        for line_name, overflow_array in overflow_history.items():
            intervals = find_overflow_intervals(overflow_array)
            if intervals:  # Only include lines that had overflows
                line_intervals[line_name] = intervals
                
                # Record all durations for this line
                for start, end in intervals:
                    duration = end - start + 1
                    all_line_durations[line_name].append(duration)
        
        # Print overflow intervals for each line
        for line_name, intervals in line_intervals.items():
            print(f"\n{line_name}: ", end="")
            for i, (start, end) in enumerate(intervals):
                if i > 0:
                    print("  ", end="")
                duration = end - start + 1
                print(f"[{start} - {end}]", end="")
            
            print("\n" + " " * len(line_name) + "  ", end="")
            
            for i, (start, end) in enumerate(intervals):
                if i > 0:
                    print("  ", end="")
                duration = end - start + 1
                print(f"{duration} steps", end="")
        
        if not line_intervals:
            print("No lines experienced overflow in this episode.")
        
        print("\n")
        
        # Create heatmap visualization
        # Filter out lines that didn't have overflows
        filtered_lines = list(line_intervals.keys())
        
        if filtered_lines:
            # Convert to DataFrame for visualization - but only include lines that had overflows
            df = pd.DataFrame({line: overflow_history[line] for line in filtered_lines})
            df['timestep'] = range(len(df))
            
            # Melt the dataframe for plotly
            df_melted = pd.melt(df, id_vars=['timestep'], value_vars=filtered_lines, 
                               var_name='Line', value_name='Overflow')
            
            # Create the heatmap
            fig = px.density_heatmap(df_melted, x='timestep', y='Line', z='Overflow',
                                  title=f'Line Overflow Heatmap - Episode {episode+1}',
                                  labels={'timestep': 'Timestep', 'Line': 'Line Name', 'Overflow': 'Overflow Status'})
            
            # Update layout for dark gray background
            fig.update_layout(
                plot_bgcolor='#333333',
                paper_bgcolor='#333333',
                font_color='white',
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            # Show plot
            print(f"HEATMAP EP {episode+1}")
            fig.show()
        else:
            print(f"HEATMAP EP {episode+1} - No overflows to display")
    
    # After all episodes, create statistics about overflow durations
    if all_line_durations:
        print("\n======= Overflow Duration Statistics ================")
        
        # Calculate maximum duration across all lines
        max_duration = max(max(durations) for durations in all_line_durations.values())
        
        # Create a bar chart for each line showing frequency of each duration
        for line_name, durations in all_line_durations.items():
            # Count frequency of each duration
            duration_counts = defaultdict(int)
            for duration in durations:
                duration_counts[duration] += 1
            
            # Convert to dataframe for plotting
            df_durations = pd.DataFrame({
                'Duration': list(duration_counts.keys()),
                'Frequency': list(duration_counts.values())
            })
            df_durations = df_durations.sort_values('Duration')
            
            # Create bar chart
            fig = px.bar(df_durations, x='Duration', y='Frequency',
                      title=f'Overflow Duration Frequencies - {line_name}',
                      labels={'Duration': 'Overflow Duration (timesteps)', 'Frequency': 'Number of Occurrences'})
            
            fig.update_layout(
                plot_bgcolor='#333333',
                paper_bgcolor='#333333',
                font_color='white',
                xaxis=dict(range=[0, max_duration+1]),
                bargap=0.2
            )
            
            # Show plot
            fig.show()
    else:
        print("No overflow statistics to display - no lines experienced overflow in any episode.")

if __name__ == "__main__":
    # Set the number of episodes to run
    nb_episodes = 2
    play_episodes_and_analyze(nb_episodes)
