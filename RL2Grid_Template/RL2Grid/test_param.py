
import grid2op
from grid2op.Parameters import Parameters
from grid2op.Agent import DoNothingAgent
from grid2op.gym_compat import GymEnv
from lightsim2grid import LightSimBackend
from grid2op.Reward import CombinedReward, IncreasingFlatReward, DistanceReward
from common.reward import LineMarginReward, RedispRewardv1

# Create the environment (using bus14 as a simple example)
print("Creating environment...")
g2op_env = grid2op.make(
    "l2rpn_case14_sandbox",
    reward_class=CombinedReward,
    backend=LightSimBackend()
)

# Setup rewards
cr = g2op_env.get_reward_instance() 
cr.addReward("IncreasingFlatReward", 
            IncreasingFlatReward(per_timestep=1/g2op_env.chronics_handler.max_episode_duration()),
            0.1)
cr.addReward("TopologyReward", DistanceReward(), 0.3)
cr.addReward("redispatchReward", RedispRewardv1(), 0.3)
cr.addReward("LineMarginReward", LineMarginReward(), 0.3)
cr.initialize(g2op_env)

# Wrap in GymEnv
gym_env = GymEnv(g2op_env, shuffle_chronics=True)

# Set parameters directly from the environment's parameters object (level 1 settings)
p = gym_env.init_env.parameters
p.HARD_OVERFLOW_THRESHOLD = 999
p.NB_TIMESTEP_OVERFLOW_ALLOWED = 9999999
p.SOFT_OVERFLOW_THRESHOLD = 1.0
p.NO_OVERFLOW_DISCONNECTION = False
p.MAX_LINE_STATUS_CHANGED = 1
p.MAX_SUB_CHANGED = 1

# Apply the modified parameters
gym_env.init_env.change_parameters(p)

# Print parameters being tested
print("Testing with parameters:")
print(f"  HARD_OVERFLOW_THRESHOLD: {p.HARD_OVERFLOW_THRESHOLD}")
print(f"  NB_TIMESTEP_OVERFLOW_ALLOWED: {p.NB_TIMESTEP_OVERFLOW_ALLOWED}")
print(f"  SOFT_OVERFLOW_THRESHOLD: {p.SOFT_OVERFLOW_THRESHOLD}")
print(f"  NO_OVERFLOW_DISCONNECTION: {p.NO_OVERFLOW_DISCONNECTION}")

# The reset is required after changing parameters
gym_env.reset()

# For testing, we'll use the underlying Grid2Op environment
agent = DoNothingAgent(gym_env.init_env.action_space)

# Play episodes
n_episodes = 7
max_steps_per_episode = gym_env.init_env.chronics_handler.max_episode_duration()
print(f"Playing {n_episodes} episodes with max {max_steps_per_episode} steps each")

survival_rates = []

for ep in range(n_episodes):
    obs = gym_env.reset()
    done = False
    step = 0
    
    # Get the underlying grid2op observation
    grid2op_obs = gym_env.init_env.get_obs()
    
    while not done and step < max_steps_per_episode:
        # Generate action using the agent
        action = agent.act(grid2op_obs, None, done)
        
        # Apply action to the Grid2Op environment
        grid2op_obs, reward, done, info = gym_env.init_env.step(action)
        step += 1
    
    survival_rate = step / max_steps_per_episode * 100
    survival_rates.append(survival_rate)
    print(f"Episode {ep+1}: Survived {step}/{max_steps_per_episode} steps ({survival_rate:.2f}%)")

# Print overall results
avg_survival = sum(survival_rates) / len(survival_rates)
print(f"\nAverage survival rate over {n_episodes} episodes: {avg_survival:.2f}%")

