from time import time

from .agent import Agent
from .config import get_alg_args
from common.imports import ap, nn, np, optim, th
from common.logger import Logger

class PPO:
    def __init__(self, envs, max_steps, run_name, start_time, args, ckpt):        
        if not ckpt.resumed: args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        batch_size = int(args.n_envs * args.n_steps)
        minibatch_size = int(batch_size // args.n_minibatches)
        n_rollouts = args.total_timesteps // batch_size
        init_rollout = 1 if not ckpt.resumed else ckpt.loaded_run['last_rollout']

        track = args.track
        if track: logger = Logger(run_name, args)
        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        continuous_actions = True if args.action_type == "redispatch" else False
        agent = Agent(envs, args, continuous_actions).to(device)

        if ckpt.resumed:
            agent.actor.load_state_dict(ckpt.loaded_run['actor'])
            agent.critic.load_state_dict(ckpt.loaded_run['critic'])

        actor_params = list(agent.actor.parameters())
        if continuous_actions: actor_params + [agent.logstd]
        ''' 
        if continuous_actions:
            actor_params.append(agent.logstd) # is this a bug? Correctly add logstd to the parameters to be optimized
        ''' 
        actor_optim = optim.Adam(actor_params, lr=args.actor_lr, eps=1e-5)
        critic_optim = optim.Adam(agent.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        if ckpt.resumed:
            actor_optim.load_state_dict(ckpt.loaded_run['actor_optim'])
            critic_optim.load_state_dict(ckpt.loaded_run['critic_optim'])

        # ALGO Logic: Storage setup
        observations = th.zeros((args.n_steps, args.n_envs) + envs.single_observation_space.shape).to(device)
        actions = th.zeros((args.n_steps, args.n_envs) + 
                           envs.single_action_space.shape ).to(device)
        logprobs = th.zeros((args.n_steps, args.n_envs)).to(device)
        rewards = th.zeros((args.n_steps, args.n_envs)).to(device)
        dones = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        values = th.zeros((args.n_steps, args.n_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0 if not ckpt.resumed else ckpt.loaded_run['global_step']
        start_time = start_time
        next_obs, info = envs.reset()
        next_obs = th.tensor(next_obs).to(device)
        next_done = th.zeros(args.n_envs).to(device)

        for iteration in range(init_rollout, n_rollouts + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / n_rollouts
                actor_optim.param_groups[0]['lr'] = frac * args.actor_lr
                critic_optim.param_groups[0]['lr'] = frac * args.critic_lr

            for step in range(0, args.n_steps):
                global_step += args.n_envs
                observations[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with th.no_grad():
                    action, logprob, _ = agent.get_action(next_obs)
                    value = agent.get_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

  
                # TRY NOT TO MODIFY: execute the game and log data.
                try:
                    print("------------------------------")
                    print("Action:", action.cpu().numpy())       ##PP
                    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                    print("Reward:", reward, global_step)    
                    #print([f"{r:.4f}" for r in reward])         ##PP
                    #print("next_obs:", next_obs)               ##PP
                    rho_values = np.array([round(x, 2) for x in next_obs[0][55:75]])
                    print("Rho values:", rho_values)
                    #print("Rho values:", [f"{r:.2f}" for r in rho_values])
                    #print("terminations:", terminations)       ##PP
                    #print("truncations:", truncations)          ##PP
                    #print("infos:", infos)                       ##PP
                    # Only print exception if it exists in final_info
                    if 'final_info' in infos and infos['final_info'] is not None:
                        for info in infos['final_info']:
                            if info and 'exception' in info:
                                print("Exception:", info['exception'])
                    '''
                    infos is always like this:
                    infos: {'final_observation': array([array([517.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -1.,  -1.,  -1.,
               -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,
               -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,
               -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,
               -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,
               -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.],
             dtype=float32)                                                    ],
      dtype=object), '_final_observation': array([ True]), 'final_info': array([{'disc_lines': array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              -1, -1, -1]), 'is_illegal': False, 'is_ambiguous': False, 'is_dispatching_illegal': False, 'is_illegal_reco': False, 'reason_alarm_illegal': None, 'reason_alert_illegal': None, 'opponent_attack_line': None, 'opponent_attack_sub': None, 'opponent_attack_duration': 0, 'exception': [Grid2OpException BackendError BackendError('Divergence of DC powerflow (non connected grid) at the initialization of AC powerflow. Detailed error: ErrorType.SolverSolve')], 'time_series_id': 'C:\\Users\\admin\\data_grid2op\\l2rpn_case14_sandbox_train\\chronics\\0829', 'rewards': {}, 'episode': {'l': [517], 'r': [1.653963565826416]}}],
      dtype=object), '_final_info': array([ True])}




                    this is next_obs, 
                    [[  1.          86.4         85.1          3.3          0.
    0.          86.87001     -1.6918808   -4.817273   -10.137186
  -10.137186    -9.145282     0.           1.           1.
    1.           1.           1.           1.           1.
    1.           1.           1.           1.           1.
    1.           1.           1.           1.           1.
    1.           1.           1.          23.2         90.1
   47.6          7.2         12.          29.5          9.3
    3.5          5.8         13.1         15.4         -1.6918808
   -4.817273    -6.5112042   -4.7619767  -10.137186   -10.4845295
  -10.683826   -10.500909   -10.778994   -10.806669   -11.431918
    0.38441375   0.35148522   0.31405112   0.32421306   0.7505744
    0.3594117    0.22512984   0.57502663   0.5250146    0.7758381
    0.26508304   0.3921261    0.31850535   0.45990786   0.43199277
    0.5405013    0.63376886   1.0293176    0.5004717    0.41924238
    0.           0.           0.           0.           0.
    0.           0.           0.           0.           0.
    0.           0.           0.           0.           0.
    0.           0.           0.           0.           0.
    0.           0.           0.           0.           0.
    0.           0.           0.           0.           0.
    0.           0.           0.           0.           0.
    0.           0.           0.           0.           0.
    0.           0.           0.           0.           0.
    0.           0.           0.           0.           0.
    0.           1.           0.           0.           1.
    1.           1.           1.           1.           1.
    1.           1.           1.           1.           1.
    1.           1.           1.           1.           2.
    1.           2.           1.           1.           1.
    1.           1.           1.           1.           1.
    1.           1.           1.           1.           1.
    1.           1.           1.           1.           1.
    1.           1.           1.           1.           1.
    1.           1.           1.           1.           1.
    1.           1.           1.           1.           1.
    1.           1.           1.           1.           1.
    1.        ]]
    rho is :  0.38441375   0.35148522   0.31405112   0.32421306   0.7505744
    0.3594117    0.22512984   0.57502663   0.5250146    0.7758381
    0.26508304   0.3921261    0.31850535   0.45990786   0.43199277
    0.5405013    0.63376886   1.0293176    0.5004717    0.41924238

    
                    '''


                except:
                    ckpt.set_record(args, agent.actor, agent.critic, global_step, actor_optim, critic_optim, logger.wb_path, iteration)
                    ckpt.save()
                    #envs.close()
                    if track: logger.close()

                next_done = np.logical_or(terminations, truncations)
                rewards[step] = th.tensor(reward).to(device).view(-1)
                next_obs, next_done = th.tensor(next_obs).to(device), th.tensor(next_done).to(device)

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                if "final_info" in infos:
                    for info in infos['final_info']:
                        if info and "episode" in info:
                            survival = info['episode']['l'][0]/max_steps
                            _return = info['episode']['r'][0]
                            if args.verbose: print(f"GLB steps={global_step}, lnth={info['episode']['l'][0]}, S_RATE={survival*100:.0f}%, RETURN={_return:.1f}")
                            print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                            if track: logger.store_metrics(survival, info)
                            break

                if track and global_step % logger.log_freq == 0: logger.log_metrics(global_step)

            # bootstrap value if not done
            with th.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = th.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.n_steps)):
                    if t == args.n_steps - 1:
                        nextnonterminal = ~next_done    # instead of (1 - next_done) just flip it
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]

                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = observations.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []
            for _ in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy = agent.get_action(b_obs[mb_inds], b_actions.long()[mb_inds])
                    newvalue = agent.get_value(b_obs[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with th.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        #old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = th.max(pg_loss1, pg_loss2).mean()

                    entropy_loss = entropy.mean()
                    pg_loss = pg_loss - args.entropy_coef * entropy_loss

                    actor_optim.zero_grad()
                    pg_loss.backward()
                    nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                    actor_optim.step()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vfloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + th.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    v_loss *= args.vf_coef

                    critic_optim.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                    critic_optim.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            #y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            #var_y = np.var(y_true)
            #explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            steps_per_second =  int(global_step / (time() - start_time))
            if args.verbose: print(f"SPS={steps_per_second}")
            '''
            if track:
                writer.add_scalar("charts/actor_learning_rate", actor_optim.param_groups[0]['lr'], global_step)
                writer.add_scalar("charts/critic_learning_rate", critic_optim.param_groups[0]['lr'], global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance", explained_var, global_step)
                writer.add_scalar("charts/SPS", steps_per_second, global_step)
                wandb.log({'SPS': steps_per_second}, step=global_step)
            '''
            # If we reach the node's time limit, we just exit the training loop, save metrics and ckpt
            if (time() - start_time) / 60 >= args.time_limit:
                break

        ckpt.set_record(args, agent.actor, agent.critic, global_step, actor_optim, critic_optim, logger.wb_path, iteration)
        ckpt.save()
        envs.close()
        if track: logger.close()
