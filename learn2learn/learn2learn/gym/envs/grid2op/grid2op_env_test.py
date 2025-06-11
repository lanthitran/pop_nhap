

# Test script for Grid2OpDirectionEnv with case14

from learn2learn.gym.envs.grid2op.grid2op_direction import Grid2OpDirectionEnv

def main():
    # Create a simple Grid2Op environment (case14)
    env = Grid2OpDirectionEnv(env_name="bus14", action_type="topology")
    obs = env.reset()
    print("Initial observation:", obs)
    done = False
    total_reward = 0.0
    step_count = 0
    while not done and step_count < 15:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {step_count}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")
        print()
        total_reward += reward
        step_count += 1

    if done:
        print(f"Episode length: {info['episode']['l'][0]}")
        print(f"Episode reward: {info['episode']['r'][0]}")
    
    print("Test finished.")
    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()