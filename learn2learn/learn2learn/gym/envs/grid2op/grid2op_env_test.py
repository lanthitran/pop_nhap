

# Test script for Grid2OpDirectionEnv with case14
import numpy as np

from learn2learn.gym.envs.grid2op.grid2op_direction import Grid2OpDirectionEnv




if __name__ == '__main__':
    # Test the environment with custom arguments
    env = Grid2OpDirectionEnv(
        task=None,
        env_name="bus14",
        action_type="topology",
        env_config_path="scenario.json",
        norm_obs=True,
        use_heuristic=True,
        heuristic_type="idle",
        seed=42,
        difficulty=0,
        reward_fn=["L2RPNRewardRegularized"],   # L2RPNRewardRegularized
        reward_factors=[1.0]
    )
    
    # Deactivate forecast before any environment interaction
    if hasattr(env.env, 'init_env'):
        env.env.init_env.deactivate_forecast()
    else:
        print("Warning: Could not deactivate forecast - init_env not found")

    # Test with a specific chronic ID
    test_task = {
        'chronics_id': 35,  # Use a specific chronic ID
        'weather_conditions': 'normal'
    }

    print("Testing with specific chronic ID...")
    env.set_task(test_task)
    
    #thermal_limits = env.env.init_env.get_thermal_limit()
    # env.env.init_env.change_reward()
    # Set random seed for action sampling
    np.random.seed(42)
    
    # Number of episodes to run
    num_episodes = 300
    
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action = 0  #env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            print(f"\nStep {step_count}:")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print(f"Total Return: {total_reward}")
            print(f"Done: {done}")
            
            if done:
                print(f"\nEpisode {episode + 1} finished after {step_count} steps")
                print(f"Episode length: {info['episode']['l'][0]}")
                print(f"Episode return: {info['episode']['r'][0]}")
                print(f"Info: {info}")

    env.close()