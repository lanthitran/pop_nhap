from typing import Any, Dict, List, Optional, Tuple

import dgl
import networkx as nx
from networkx.classes.function import restricted_view
from pop.configs import architecture
import psutil
from grid2op.Environment import BaseEnv
from ray.util.client import ray
from tqdm import tqdm

from pop.agents.manager import Manager
from pop.agents.ray_gcn_agent import RayGCNAgent
from pop.agents.ray_shallow_gcn_agent import RayShallowGCNAgent
from pop.community_detection.community_detector import Community
from pop.configs.architecture import Architecture
from pop.multiagent_system.base_pop import BasePOP
from pop.multiagent_system.space_factorization import EncodedAction, Substation
import random

from pop.multiagent_system.dictatorship_penalizer import DictatorshipPenalizer


""" 
DPOP (Distributed Partially Observable Planning) class implements a hierarchical multi-agent system
for power grid control using reinforcement learning. It extends BasePOP with distributed capabilities
and manages a hierarchy of agents including a head manager and community managers. | Hung |

This class implements a distributed multi-agent system for power grid control using reinforcement learning.
Key components:
 - Head Manager: Top-level agent that coordinates community managers
 - Community Managers: Mid-level agents that manage groups of substations
 - Substation Agents: Low-level agents that control individual substations
The system uses Ray for distributed computing and implements a hierarchical decision-making process. | Hung |
| Hung |
"""

class DPOP(BasePOP):
    def __init__(
        self,
        env: BaseEnv,  # Grid2Op environment for power grid simulation | Hung |
        name: str,     # Identifier for this DPOP instance | Hung |
        architecture: Architecture,  # Neural network architecture configuration | Hung |
        training: bool,  # Whether in training or evaluation mode | Hung |
        seed: int,     # Random seed for reproducibility | Hung |
        feature_ranges: Dict[str, Tuple[float, float]],  # Valid ranges for features | Hung |
        checkpoint_dir: Optional[str] = None,  # Directory to save model checkpoints | Hung |
        tensorboard_dir: Optional[str] = None,  # Directory for tensorboard logs | Hung |
        device: Optional[str] = None,  # Computing device (CPU/GPU) | Hung |
        local: bool = False,  # Whether to run in local mode | Hung |
        pre_train: bool = False,  # Whether in pre-training phase | Hung |
    ):
        # Set CPU affinity to specific cores for better performance | Hung |
        process = psutil.Process()
        process.cpu_affinity(list(range(0, 14)))
        print("Running on cores: " + str(process.cpu_affinity()))
        
        # Initialize Ray for distributed computing | Hung |
        ray.init(local_mode=local, num_cpus=len(process.cpu_affinity()) * 2)
        
        # Initialize parent class with basic configuration | Hung |
        super(DPOP, self).__init__(
            env=env,
            name=name,
            architecture=architecture,
            training=training,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            device=device,
            pre_train=pre_train,
            feature_ranges=feature_ranges,
        )

        # Extract node features from architecture configuration | Hung |
        # This determines the size of neural network layers | Hung |
        try:
            node_features = self.architecture.manager.embedding.layers[-1].kwargs[
                "out_feats"
            ]
        except KeyError:
            # If last layer is an activation one
            # Should make this general
            # If last layer is an activation one, get features from second-to-last layer | Hung |
            node_features = self.architecture.manager.embedding.layers[-2].kwargs[
                "out_feats"
            ]

        # Define feature ranges for different components | Hung |
        # These ranges are used for normalization and validation | Hung |
        sub_ranges = {
            "sub_" + str(n_sub): tuple([0, 1]) for n_sub in range(self.env.n_sub * 2)
        }
        embedding_ranges = {
            "embedding_" + str(feat): tuple([0, 1])
            for feat in range(int(node_features))
        }
        action_ranges = {
            "action": self.manager_feature_ranges["node_features"]["action"]
        }

        # Head Manager Initialization
        # Creates the top-level manager with remote capabilities using Ray | Hung |
        self.head_manager: Optional[Manager] = Manager.remote(
            agent_actions=self.env.n_sub * 2,
            node_features=[self.architecture.pop.head_manager_embedding_name],
            architecture=self.architecture.head_manager,
            name="head_manager_" + self.name,
            training=self.training,
            device=str(self.device),
            single_node_features=int(node_features)
            + self.env.n_sub * 2
            + 1,  # Manager Node Embedding + Manager Community (1 hot encoded) + selected action
            feature_ranges={
                "node_features": {**sub_ranges, **embedding_ranges, **action_ranges}
            },
        )

        # Initialize dictatorship penalty if configured | Hung |
        # This helps prevent any single agent from dominating decision-making | Hung |
        if self.architecture.pop.dictatorship_penalty:
            self.dictatorship_penalty = DictatorshipPenalizer(
                {choice: 1 for choice in range(self.env.n_sub)},
                **self.architecture.pop.dictatorship_penalty
            )

    def get_action(self, graph: dgl.DGLHeteroGraph) -> Tuple[int, Optional[float]]:
        # Get action from head manager or random action during pre-training | Hung |
        # During pre-training, actions are random to encourage exploration | Hung |
        if self.pre_train:
            return random.sample(list(range(graph.num_nodes())), 1)[0], None
        else:
            # Filters out invalid actions based on current state | Hung |
            mask = [
                node
                for node in range(graph.num_nodes())
                if not self.architecture.pop.manager_remove_no_action
                or graph.ndata["embedding_community_action"][node][-1].item() != 0
            ]

            # Get action from head manager using Ray for distributed execution | Hung |
            chosen_node, q_value = ray.get(
                self.head_manager.take_action.remote(graph, mask=mask)
            )

        # Log exploration metrics | Hung |
        self.log_exploration(
            "head_manager",
            ray.get(self.head_manager.get_exploration_logs.remote()),
            self.train_steps,
        )

        return int(chosen_node), q_value

    def _extra_step(
        self,
        action: int,  # Selected action | Hung |
        reward: float,  # Reward received | Hung |
        next_sub_graphs: Dict[Community, dgl.DGLHeteroGraph],  # Next state sub-graphs | Hung |
        next_substation_to_encoded_action: Dict[Substation, EncodedAction],  # Next actions | Hung |
        next_graph: nx.Graph,  # Next state graph | Hung |
        done: bool,  # Whether episode is done | Hung |
        next_communities: List[Community],  # Next state communities | Hung |
        next_community_to_manager: Dict[Community, Manager],  # Next state managers | Hung |
    ):
        # Calculate dictatorship penalty if enabled | Hung |
        # Penalizes actions that give too much control to a single agent | Hung |
        penalty = 0
        if self.architecture.pop.dictatorship_penalty:
            penalty = self.dictatorship_penalty.penalty(action)

        if done:
            # Handle terminal state | Hung |
            # Updates the head manager with final state information | Hung |
            loss, full_reward = ray.get(
                self.head_manager.step.remote(
                    observation=self.summarized_graph,
                    action=action,
                    reward=reward + penalty,
                    next_observation=dgl.DGLGraph(),
                    done=done,
                    stop_decay=False,
                )
            )
        else:
            # Get next state information | Hung |
            # Computes the next state representation for the head manager | Hung |
            next_community_to_substation, _ = self._get_manager_actions(
                next_graph,
                next_sub_graphs,
                next_substation_to_encoded_action,
                next_communities,
                next_community_to_manager,
            )

            # Compute summarized graph for next state | Hung |
            # Creates a compact representation of the next state | Hung |
            next_summarized_graph: dgl.DGLHeteroGraph = self._compute_summarized_graph(
                next_graph,
                next_sub_graphs,
                next_substation_to_encoded_action,
                next_community_to_substation,
                new_communities=next_communities,
                new_community_to_manager_dict=next_community_to_manager,
            )

            # Update head manager with new state information | Hung |
            loss, full_reward = ray.get(
                self.head_manager.step.remote(
                    observation=self.summarized_graph,
                    action=action,
                    reward=reward + penalty,
                    next_observation=next_summarized_graph,
                    done=done,
                    stop_decay=False,
                )
            )

        # Log training metrics | Hung |
        self.log_step(
            losses=[loss],
            implicit_rewards=[full_reward - reward],
            names=["Head Manager"],
            train_steps=self.train_steps,
            dictatorship_penalties=[penalty],
        )

    def get_state(self: "DPOP") -> Dict[str, Any]:
        # Get current state including head manager state | Hung |
        # Used for saving checkpoints and resuming training | Hung |
        state: Dict[str, Any] = super().get_state()
        state["head_manager_state"] = ray.get(self.head_manager.get_state.remote())
        return state

    @staticmethod
    def factory(
        checkpoint: Dict[str, Any],  # Checkpoint data to load | Hung |
        env: Optional[BaseEnv] = None,  # Environment instance | Hung |
        tensorboard_dir: Optional[str] = None,  # Tensorboard directory | Hung |
        checkpoint_dir: Optional[str] = None,  # Checkpoint directory | Hung |
        name: Optional[str] = None,  # Instance name | Hung |
        training: Optional[bool] = None,  # Training mode | Hung |
        local: bool = False,  # Local mode flag | Hung |
        pre_train: bool = False,  # Pre-training flag | Hung |
        reset_exploration: bool = False,  # Reset exploration flag | Hung |
        architecture: Optional[Architecture] = None,  # Architecture configuration | Hung |
    ) -> "DPOP":
        # Create DPOP instance from checkpoint | Hung |
        dpop: "DPOP" = DPOP(
            env=env,
            name=checkpoint["name"] if name is None else name,
            architecture=Architecture(load_from_dict=checkpoint["architecture"])
            if architecture is None
            else architecture,
            training=training,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            seed=checkpoint["seed"],
            device=checkpoint["device"],
            local=local,
            pre_train=pre_train,
            feature_ranges=checkpoint["feature_ranges"],
        )
        dpop.pre_initialized = True
        dpop.alive_steps = checkpoint["alive_steps"] if training else 0
        dpop.episodes = checkpoint["episodes"] if training else 0
        dpop.train_steps = checkpoint["train_steps"] if training else 0
        dpop.edge_features = checkpoint["edge_features"]
        dpop.node_features = checkpoint["node_features"]
        dpop.manager_initialization_threshold = checkpoint[
            "manager_initialization_threshold"
        ]
        dpop.community_update_steps = checkpoint["community_update_steps"]
        dpop.substation_to_agent = {
            sub_id: RayGCNAgent.load(
                checkpoint=agent_state,
                training=training,
                reset_exploration=reset_exploration,
                architecture=architecture.agent if architecture is not None else None,
            )
            if "optimizer_state" in list(agent_state.keys())
            else RayShallowGCNAgent.load(checkpoint=agent_state)
            for sub_id, agent_state in tqdm(checkpoint["agents_state"].items())
        }
        print("Loading Managers")
        dpop.managers_history = {
            Manager.load(
                checkpoint=manager_state,
                training=training,
                reset_exploration=reset_exploration,
                architecture=architecture.manager if architecture is not None else None,
            ): history
            for _, (manager_state, history) in tqdm(
                checkpoint["managers_state"].items()
            )
        }
        # TODO: managers here are treated as equal, they may be not

        if dpop.architecture.pop.incentives:
            for manager in dpop.managers_history.keys():
                dpop.manager_incentives.add_agent(manager, 1)

        if dpop.architecture.pop.dictatorship_penalty:
            for manager in dpop.managers_history.keys():
                dpop.manager_dictatorship_penalties[manager] = DictatorshipPenalizer(
                    choice_to_ranking={
                        substation: len(action_converter.all_actions)
                        for substation, action_converter in dpop.substation_to_action_converter.items()
                    },
                    **dpop.architecture.pop.dictatorship_penalty
                )

        print("Loading Head Manager")
        dpop.head_manager = Manager.load(
            checkpoint=checkpoint["head_manager_state"],
            training=training,
            reset_exploration=reset_exploration,
            architecture=architecture.head_manager
            if architecture is not None
            else None,
        )
        print("Loading is over")
        return dpop
