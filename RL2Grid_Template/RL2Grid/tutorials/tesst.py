from grid2op.gym_compat import GymEnv
import grid2op
from gymnasium import Env
from gymnasium.utils.env_checker import check_env
try:
    from lightsim2grid import LightSimBackend
    bk_cls = LightSimBackend
except ImportError as exc:
    print(f"Error: {exc} when importing faster LightSimBackend")
    from grid2op.Backend import PandaPowerBackend
    bk_cls = PandaPowerBackend
    
# l2rpn_case14_sandbox   or  educ_case14_storage

env_name = "educ_case14_storage" 
training_env = grid2op.make(env_name, test=True, backend=bk_cls())  # we put "test=True" in this notebook because...
# it's a notebook to explain things. Of course, do not put "test=True" if you really want
# to train an agent...
gym_env = GymEnv(training_env)

isinstance(gym_env, Env)

check_env(gym_env, warn=False)

from grid2op.gym_compat import DiscreteActSpace
gym_env.action_space = DiscreteActSpace(training_env.action_space,
                                        attr_to_keep=["set_bus" , "set_line_status_simple"])
gym_env.action_space

from grid2op.gym_compat import BoxGymObsSpace
gym_env.observation_space = BoxGymObsSpace(training_env.observation_space,
                                           attr_to_keep=["rho"])
gym_env.observation_space


from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=f"./checkpoints_test",
    name_prefix="test_model"
)
n_envs = 2

from stable_baselines3 import PPO
#this one works?

if 1:
        
    n_steps = 20000

    model = PPO(env=gym_env,
                learning_rate=1e-3,
                policy="MlpPolicy",
                
                n_steps=n_steps,
                batch_size=(20000 * n_envs) // 1563, 
                n_epochs=40,                     # From config.py: update_epochs
            gamma=0.9,                       # From config.py: gamma
            gae_lambda=0.95,                 # From config.py: gae_lambda
            clip_range=0.2,   
                    normalize_advantage=True,        # From config.py: norm_adv
            ent_coef=0.01,                   # From config.py: entropy_coef
            vf_coef=0.5,                     # From config.py: vf_coef
            max_grad_norm=10,  
            verbose=True,  
                    
                tensorboard_log="./tesst/",

                policy_kwargs={
                "net_arch": {
                    "pi": [256, 256],   # From config.py: actor_layers
                    "vf": [256, 256]    # From config.py: critic_layers
                }
            }
                #,zzzz=2 
                #device = "cpu"
                )
    print(f"Training PPO agent for  timesteps...")
    model.learn(total_timesteps=1000000, callback=checkpoint_callback)




else:
        


    #this one doesn't work? why?
    n_steps = 20000  # From config.py, no longer scaling with n_envs

    # Create PPO model with parameters matching config.py
    model = PPO(
        policy="MlpPolicy",
        env=gym_env, #envs  env_simple
        learning_rate=2.5e-4,            # From config.py: actor_lr and critic_lr
        n_steps=n_steps,                 # From config.py: 20000 (constant regardless of env count)
        batch_size=(20000 * n_envs) // 1563,  # Scale with both n_steps and n_envs (n_minibatches=4)
        n_epochs=40,                     # From config.py: update_epochs
        gamma=0.9,                       # From config.py: gamma
        gae_lambda=0.95,                 # From config.py: gae_lambda
        clip_range=0.2,                  # From config.py: clip_coef
        normalize_advantage=True,        # From config.py: norm_adv
        ent_coef=0.01,                   # From config.py: entropy_coef
        vf_coef=0.5,                     # From config.py: vf_coef
        max_grad_norm=10,                # From config.py: max_grad_norm
        verbose=True,                  # Verbose level (0, 1, or 2)
        tensorboard_log="./ppo_STB_BL_tensorboard/",  # TensorBoard log directory
        policy_kwargs={
            "net_arch": {
                "pi": [256, 256],   # From config.py: actor_layers
                "vf": [256, 256]    # From config.py: critic_layers
            }
        }
    )

    # Train model
    model.learn(total_timesteps=8888, callback=checkpoint_callback)
