#!/usr/bin/env python3

import cherry as ch
from cherry._utils import _min_size, _istensorable
from cherry.envs.utils import is_vectorized
from .base_wrapper import Wrapper

from collections.abc import Iterable


def flatten_episodes(replay, episodes, num_workers):
    """
    input might be:
    
    # Replay collected by Runner (list of batched Transitions)
    replay = [
        # Time step 0 (across workers)
        Transition(
            state=[s_W0_ep1_t0, s_W1_ep1_t0],
            action=[a_W0_ep1_t0, a_W1_ep1_t0],
            # ... other fields ...
            done=[False, False],
            info=[info_W0_ep1_t0, info_W1_ep1_t0]
        ),
        # Time step 1 (across workers)
        Transition(
            state=[s_W0_ep1_t1, s_W1_ep1_t1],
            action=[a_W0_ep1_t1, a_W1_ep1_t1],
            # ... other fields ...
            done=[True, False], # Worker 0 finished its 1st episode!
            info=[info_W0_ep1_t1, info_W1_ep1_t1]
        ),
        # Time step 2 (across workers)
        Transition(
            state=[s_W0_ep2_t0, s_W1_ep1_t2], # Worker 0 started its 2nd episode
            action=[a_W0_ep2_t0, a_W1_ep1_t2],
            # ... other fields ...
            done=[True, True], # Worker 0 finished its 2nd ep (length 1), Worker 1 finished its 1st ep (length 3)
            info=[info_W0_ep2_t0, info_W1_ep1_t2]
        ),
        # ...
    ]

    output might be:
    # flat_replay returned by flatten_episodes (list of individual Transitions)
    flat_replay = [
        # Worker 0's 1st episode (length 2) - Completed at sars_t1
        Transition(state=s_W0_ep1_t0, action=a_W0_ep1_t0, reward=r_W0_ep1_t0, next_state=ns_W0_ep1_t0, done=False, runner_id=0, info=info_W0_ep1_t0),
        Transition(state=s_W0_ep1_t1, action=a_W0_ep1_t1, reward=r_W0_ep1_t1, next_state=ns_W0_ep1_t1, done=True,  runner_id=0, info=info_W0_ep1_t1),

        # Worker 0's 2nd episode (length 1) - Completed at sars_t2 (processed first for worker 0 in sars_t2)
        Transition(state=s_W0_ep2_t0, action=a_W0_ep2_t0, reward=r_W0_ep2_t0, next_state=ns_W0_ep2_t0, done=True,  runner_id=0, info=info_W0_ep2_t0),

        # Worker 1's 1st episode (length 3) - Completed at sars_t2 (processed after worker 0 for sars_t2)
        Transition(state=s_W1_ep1_t0, action=a_W1_ep1_t0, reward=r_W1_ep1_t0, next_state=ns_W1_ep1_t0, done=False, runner_id=1, info=info_W1_ep1_t0),
        Transition(state=s_W1_ep1_t1, action=a_W1_ep1_t1, reward=r_W1_ep1_t1, next_state=ns_W1_ep1_t1, done=False, runner_id=1, info=info_W1_ep1_t1),
        Transition(state=s_W1_ep1_t2, action=a_W1_ep1_t2, reward=r_W1_ep1_t2, next_state=ns_W1_ep1_t2, done=True,  runner_id=1, info=info_W1_ep1_t2),
    ]


    """
    #  TODO: This implementation is not efficient.

    #  NOTE: Additional info (other than a transition's default fields) is simply copied.
    #  To know from which worker the data was gathered, you can access sars.runner_id
    #  TODO: This is not great. What is the best behaviour with infos here ?
    flat_replay = ch.ExperienceReplay()
    worker_replays = [ch.ExperienceReplay() for w in range(num_workers)]
    flat_episodes = 0
    
    for sars in replay:
        # Reshape tensors to handle batched data        | Hung |
        state = sars.state.view(_min_size(sars.state))
        action = sars.action.view(_min_size(sars.action))
        reward = sars.reward.view(_min_size(sars.reward))
        next_state = sars.next_state.view(_min_size(sars.next_state))
        done = sars.done.view(_min_size(sars.done))
        
        # Get additional info fields       | Hung |
        fields = set(sars._fields) - {'state', 'action', 'reward', 'next_state', 'done'}
        infos = {f: getattr(sars, f) for f in fields}
        
        # Process each worker's data       | Hung |
        for worker in range(num_workers):
            infos['runner_id'] = worker
            # The following attemps to split additional infos. (WIP. Remove ?)
            # infos = {}
            # for f in fields:
            #     value = getattr(sars, f)
            #     if isinstance(value, Iterable) and len(value) == num_workers:
            #         value = value[worker]
            #     elif _istensorable(value):
            #         tvalue = ch.totensor(value)
            #         tvalue = tvalue.view(_min_size(tvalue))
            #         if tvalue.size(0) == num_workers:
            #             value = tvalue[worker]
            #     infos[f] = value
            worker_replays[worker].append(state[worker],
                                          action[worker],
                                          reward[worker],
                                          next_state[worker],
                                          done[worker],
                                          **infos,
                                          )
            # If episode is done, merge worker's replay into flat replay      | Hung |
            if bool(done[worker]):
                flat_replay += worker_replays[worker]
                worker_replays[worker] = ch.ExperienceReplay()
                flat_episodes += 1
                
            # Check if we've collected enough episodes        | Hung |
            if flat_episodes >= episodes:
                break
                
        if flat_episodes >= episodes:
            break
            
    return flat_replay


class Runner(Wrapper):

    """
    <a href="" class="source-link">[Source]</a>

    ## Description

    Helps collect transitions, given a `get_action` function.

    ## Example

    ~~~python
    env = MyEnv()
    env = Runner(env)
    replay = env.run(lambda x: policy(x), steps=100)
    # or
    replay = env.run(lambda x: policy(x), episodes=5)
    ~~~

    """
    #  TODO: When is_vectorized and using episodes=n, use the parallel
    #  environmnents to sample n episodes, and stack them inside a flat replay.

    def __init__(self, env):
        super(Runner, self).__init__(env)
        self.env = env
        self._needs_reset = True
        self._current_state = None

    def reset(self, *args, **kwargs):
        self._current_state = self.env.reset(*args, **kwargs)
        self._needs_reset = False
        return self._current_state

    def step(self, action, *args, **kwargs):
        # TODO: Implement it to be compatible with .run()
        raise NotImplementedError('Runner does not currently support step.')

    def run(self,
            get_action,
            steps=None,
            episodes=None,
            render=False):
        """
        ## Description

        Runner wrapper's run method.

        !!! info
            Either use the `steps` OR the `episodes` argument.

        ## Arguments

        * `get_action` (function) - Given a state, returns the action to be taken.
        * `steps` (int, *optional*, default=None) - The number of steps to be collected.
        * `episodes` (int, *optional*, default=None) - The number of episodes to be collected.
        """

        if steps is None:
            steps = float('inf')
            if self.is_vectorized:
                self._needs_reset = True
        elif episodes is None:
            episodes = float('inf')
        else:
            msg = 'Either steps or episodes should be set.'
            raise Exception(msg)

        replay = ch.ExperienceReplay(vectorized=self.is_vectorized)
        collected_episodes = 0
        collected_steps = 0
        while True:
            if collected_steps >= steps or collected_episodes >= episodes:
                if self.is_vectorized and collected_episodes >= episodes:
                    replay = flatten_episodes(replay, episodes, self.num_envs)
                    self._needs_reset = True
                return replay
            if self._needs_reset:
                self.reset()
            info = {}
            action = get_action(self._current_state)
            if isinstance(action, tuple):
                skip_unpack = False
                if self.is_vectorized:
                    if len(action) > 2:
                        skip_unpack = True
                    elif len(action) == 2 and \
                            self.env.num_envs == 2 and \
                            not isinstance(action[1], dict):
                        # action[1] is not info but an action
                        action = (action, )

                if not skip_unpack:
                    if len(action) == 2:
                        info = action[1]
                        action = action[0]
                    elif len(action) == 1:
                        action = action[0]
                    else:
                        msg = 'get_action should return 1 or 2 values.'
                        raise NotImplementedError(msg)
            old_state = self._current_state
            state, reward, done, _ = self.env.step(action)
            if not self.is_vectorized and done:
                collected_episodes += 1
                self._needs_reset = True
            elif self.is_vectorized:
                collected_episodes += sum(done)
            replay.append(old_state, action, reward, state, done, **info)
            self._current_state = state
            if render:
                self.env.render()
            collected_steps += 1
