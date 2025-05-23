from typing import Any, Dict, Optional
from grid2op.Agent.agentWithConverter import AgentWithConverter
from grid2op.Environment.BaseEnv import BaseEnv
from grid2op.Agent.recoPowerlineAgent import RecoPowerlineAgent
from grid2op.Observation.baseObservation import BaseObservation
from pop.configs.architecture import Architecture
from pop.multiagent_system.dpop import DPOP
from pop.multiagent_system.space_factorization import EncodedAction
from pop.networks.serializable_module import SerializableModule


"""
ExpertPop implements an expert-guided multi-agent system for power grid control.
It combines expert knowledge with learned policies to make decisions about grid operations.

Key components:
- Expert knowledge for powerline reconnection
- Safe operation limits for grid parameters
- Integration with DPOP for complex decision making

The system prioritizes:
1. Powerline reconnection when possible
2. Safe operation within grid limits
3. Learned policies for complex scenarios
| Hung |
"""

class ExpertPop(SerializableModule, AgentWithConverter):
    """
    Expert-guided POP implementation that combines expert rules with learned policies.
    
    This class:
    - Manages powerline reconnection using expert rules
    - Enforces safe operation limits
    - Integrates with DPOP for complex decisions
    - Tracks expert-guided steps and episodes
    
    The system uses a hierarchical decision process:
    1. Check for powerline reconnection opportunities
    2. Monitor grid safety parameters
    3. Delegate to learned policies when needed
    | Hung |
    """
    def __init__(
        self, pop: DPOP, checkpoint_dir: str, expert_only: bool = False
    ) -> None:
        # Initialize base classes and set up expert components | Hung |
        super().__init__(log_dir=checkpoint_dir, name=pop.name)
        # Expert agent for powerline reconnection decisions | Hung |
        self.greedy_reconnect_agent = RecoPowerlineAgent(pop.env.action_space)
        # Safety thresholds from architecture config | Hung |
        self.safe_max_rho = pop.architecture.pop.safe_max_rho
        self.curtail_storage_limit = pop.architecture.pop.curtail_storage_limit
        # Track expert-guided steps and state | Hung |
        self.expert_steps = 0
        self.step_pop = False
        self.pop = pop
        self.expert_only = expert_only

    def my_act(self, observation, reward, done=False):
        """
        Main decision making method that implements expert-guided control logic.
        
        Decision process:
        1. Check for powerline reconnection opportunities
        2. Monitor grid safety parameters
        3. Delegate to learned policies if needed
        4. Take no action as fallback
        
        Returns the selected action based on expert rules and learned policies.
        | Hung |
        """
        self.expert_steps += 1
        # Get expert recommendation for powerline reconnection | Hung |
        reconnection_action = self.greedy_reconnect_agent.act(observation, reward)

        if reconnection_action.impact_on_objects()["has_impact"]:
            # If there is some powerline to reconnect do it
            self.step_dpop = False
            action = reconnection_action

        elif not self.expert_only and max(observation.rho) > self.safe_max_rho:
            # If there is some powerline overloaded ask the agent what to do
            self.step_dpop = True
            action = self.pop.act(observation, reward, done)
            # Apply safety limits to learned policy actions | Hung |
            action.limit_curtail_storage(observation, margin=self.curtail_storage_limit)

        else:
            # else do nothing
            self.step_dpop = False
            action = self.pop.env.action_space({})

        # Log expert-guided actions for monitoring | Hung |
        self.pop.writer.add_text("Expert Action", str(action), self.expert_steps)
        return action

    def convert_act(self, action):
        """
        Action conversion method - returns action unchanged.
        Maintains compatibility with AgentWithConverter interface.
        | Hung |
        """
        return action

    def convert_obs(self, observation: BaseObservation) -> BaseObservation:
        """
        Observation conversion method - returns observation unchanged.
        Maintains compatibility with AgentWithConverter interface.
        | Hung |
        """
        return observation

    def step(
        self,
        action: EncodedAction,
        observation: BaseObservation,
        reward: float,
        next_observation: BaseObservation,
        done: bool,
    ):
        """
        Process step results and update system state.
        
        Handles:
        - Reward logging
        - DPOP step updates when used
        - Episode tracking and logging
        | Hung |
        """
        # Log rewards for monitoring and analysis | Hung |
        self.pop.log_reward(reward, self.expert_steps, name="Expert Reward")

        if self.step_dpop:
            # Update DPOP when it was used for decision making | Hung |
            self.pop.step(action, observation, reward, next_observation, done)
        else:
            # Track steps and episodes when using expert rules | Hung |
            self.pop.alive_steps += 1
            if done:
                self.pop.log_alive_steps(self.pop.alive_steps, self.pop.episodes)
                self.pop.episodes += 1
                self.pop.alive_steps = 0

    def get_state(self) -> Dict[str, Any]:
        """
        Get current system state including expert steps.
        Used for checkpointing and state restoration.
        | Hung |
        """
        state: Dict[str, Any] = self.pop.get_state()
        state["expert_steps"] = self.expert_steps
        return state

    @property
    def episodes(self):
        """
        Get current episode count from DPOP.
        | Hung |
        """
        return self.pop.episodes

    @property
    def writer(self):
        """
        Get tensorboard writer from DPOP.
        | Hung |
        """
        return self.pop.writer

    @property
    def train_steps(self):
        """
        Get number of expert-guided steps.
        | Hung |
        """
        return self.expert_steps

    @staticmethod
    def factory(
        checkpoint: Dict[str, Any],
        env: Optional[BaseEnv] = None,
        tensorboard_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        name: Optional[str] = None,
        training: Optional[bool] = None,
        local: bool = False,
        pre_train: bool = False,
        reset_exploration: bool = False,
        architecture: Optional[Architecture] = None,
    ) -> "ExpertPop":
        """
        Factory method to create ExpertPop instance from checkpoint.
        
        Process:
        1. Create DPOP instance from checkpoint
        2. Initialize ExpertPop with DPOP
        3. Restore expert steps from checkpoint
        
        Returns configured ExpertPop instance.
        | Hung |
        """
        # Create DPOP instance from checkpoint data | Hung |
        pop = DPOP.factory(
            checkpoint,
            env,
            tensorboard_dir,
            checkpoint_dir,
            name,
            training,
            local,
            pre_train,
            reset_exploration,
            architecture,
        )
        # Create and configure ExpertPop instance | Hung |
        expert_pop = ExpertPop(
            pop,
            checkpoint_dir
            if checkpoint_dir is not None
            else checkpoint["checkpoint_dir"],
        )
        # Restore expert steps from checkpoint | Hung |
        expert_pop.expert_steps = checkpoint["expert_steps"]
        return expert_pop
