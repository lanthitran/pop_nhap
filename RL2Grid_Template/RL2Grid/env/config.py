from common.imports import ap
from common.utils import str2bool

def get_env_args():
    parser = ap.ArgumentParser()

    # Settings
    parser.add_argument("--env-id", type=str, default="bus14", help="ID of the grid2op environment")
    parser.add_argument("--n-envs", type=int, default=10, help="Number of parallel envs to run")
    parser.add_argument("--action-type", type=str, default="topology", choices=["topology", "redispatch"], help="Type of environment: topology (discrete) or redispatch (continuous)")
    parser.add_argument("--difficulty", type=int, default=0, help="Higher difficulty means bigger action spaces")

    # Scenarios
    parser.add_argument("--env-config-path", type=str, default="scenario.json", help="Path to environment configuration file")

    # Normalization
    parser.add_argument("--norm-obs", type=str2bool, default=True, help="Toggle normalize observations")
    parser.add_argument("--use-heuristic", type=str2bool, default=True, help="Toggles heuristics for base operations")

    # Parse the arguments
    params, _ = parser.parse_known_args()

    return params
