from common.imports import ap
from common.utils import str2bool

def get_alg_args():
    parser = ap.ArgumentParser()

    parser.add_argument("--total-timesteps", type=int, default=int(1e6), help="Total timesteps for the experiment")
    parser.add_argument("--learning-starts", type=int, default=int(2e4), help="When to start learning")

    parser.add_argument('--actor-layers', nargs='+', type=int, default=[32, 32], help='Actor network size')
    parser.add_argument('--critic-layers', nargs='+', type=int, default=[32, 32], help='Critic network size')
    parser.add_argument('--actor-act-fn', type=str, default='tanh', help='Actor activation function')
    parser.add_argument('--critic-act-fn', type=str, default='tanh', help='Critic activation function')
    parser.add_argument("--actor-lr", type=float, default=2.5e-3, help="Learning rate for the actor")
    parser.add_argument("--critic-lr", type=float, default=2.5e-3, help="Learning rate for the critic")
    parser.add_argument('--train-freq', type=int, default=20, help='Training frequency in timesteps')

    parser.add_argument("--gamma", type=float, default=.9, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target smoothing coefficient")
    parser.add_argument("--noise-clip", type=float, default=0.5, help="Noise clip for target policy smoothing regularization")
    parser.add_argument("--alpha", type=float, default=0.2, help="Entropy regularization coefficient")
    parser.add_argument("--autotune", type=str2bool, default=True, help="Automatic tuning of the entropy coefficient")

    parser.add_argument('--actor-train-freq', type=int, default=2, help='Update frequency for (delayed) actor training')
    parser.add_argument('--tg-critic-freq', type=int, default=1, help='Update frequency for target networks')

    parser.add_argument("--buffer-size", type=int, default=int(500000), help="Replay memory buffer size")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size of sample from the replay memory")

    return parser.parse_known_args()[0]

  