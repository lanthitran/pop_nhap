


import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from grid2op.Agent import DoNothingAgent

def play_episode_with_do_nothing():
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
        
        if step % 100 == 0:
            print(f"Step {step}, max_rho={obs.rho.max():.2f}")
    
    print(f"Episode completed after {step} steps")
    
    # Convert to DataFrame for visualization
    df = pd.DataFrame(overflow_history)
    df['timestep'] = range(len(df))
    
    # Melt the dataframe for plotly
    df_melted = pd.melt(df, id_vars=['timestep'], value_vars=line_names, 
                        var_name='Line', value_name='Overflow')
    
    # Create the heatmap
    fig = px.density_heatmap(df_melted, x='timestep', y='Line', z='Overflow',
                           title='Line Overflow Heatmap (1=Overflow, 0=Normal)',
                           labels={'timestep': 'Timestep', 'Line': 'Line Name', 'Overflow': 'Overflow Status'})
    
    # Update layout for gray background
    fig.update_layout(
        plot_bgcolor='gray',
        paper_bgcolor='gray',
        font_color='white'
    )
    
    # Count overflow timesteps per line
    overflow_counts = {line: sum(values) for line, values in overflow_history.items()}
    
    # Create bar chart for total overflow timesteps
    overflow_df = pd.DataFrame({
        'Line': list(overflow_counts.keys()),
        'Overflow_Timesteps': list(overflow_counts.values())
    })
    overflow_df = overflow_df.sort_values('Overflow_Timesteps', ascending=False)
    
    fig2 = px.bar(overflow_df, x='Line', y='Overflow_Timesteps',
                 title='Total Overflow Timesteps by Line',
                 labels={'Line': 'Line Name', 'Overflow_Timesteps': 'Timesteps in Overflow'})
    
    # Update layout for gray background
    fig2.update_layout(
        plot_bgcolor='gray',
        paper_bgcolor='gray',
        font_color='white'
    )
    
    # Show plots
    fig.show()
    fig2.show()
    
    return overflow_history

if __name__ == "__main__":
    overflow_history = play_episode_with_do_nothing()
