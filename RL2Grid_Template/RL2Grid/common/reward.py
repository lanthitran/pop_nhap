# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from .imports import np

from grid2op.Reward.baseReward import BaseReward
from grid2op.Reward import RedispReward
from grid2op.Reward import L2RPNReward
from grid2op.dtypes import dt_float

class LineMarginReward(BaseReward):
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        self.penalty = dt_float(-1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error:
            return self.penalty * sum(~env.current_obs.line_status) / env.current_obs.n_line
        
        if is_illegal or is_ambiguous:
            return 0.0

        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        margin = np.divide(thermal_limits - ampere_flows, thermal_limits + 1e-10)

        # Reward is based on how much lines are used (the lower the better and goes negative in case of overflow) and is penalized for disconnected lines. We then normalize everything between (more or less) [-1, 1]  
        reward = margin[env.current_obs.line_status].sum() + (self.penalty * sum(~env.current_obs.line_status))

        return reward / env.current_obs.n_line
      
class RedispRewardv1(RedispReward):
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            # if the episode is over and it's my fault (i did a blackout) i strongly
            if has_error or is_illegal or is_ambiguous:
                return self.reward_min
        elif is_illegal or is_ambiguous:
            return self._reward_illegal_ambiguous

        # compute the losses
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        # don't forget to convert MW to MWh !
        losses = (gen_p.sum() - load_p.sum()) * env.delta_time_seconds / 3600.0
        # compute the marginal cost
        gen_activeprod_t = env._gen_activeprod_t
        marginal_cost = np.max(env.gen_cost_per_MW[gen_activeprod_t > 0.0])
        # redispatching amount
        actual_dispatch = env._actual_dispatch
        redisp_cost = (
            self._alpha_redisp * np.abs(actual_dispatch).sum() * marginal_cost * env.delta_time_seconds / 3600.0
        )
        
        # cost of losses
        losses_cost = losses * marginal_cost

        # cost of storage
        c_storage = np.abs(env._storage_power).sum() * marginal_cost * env.delta_time_seconds / 3600.0
        
        # total "regret"
        regret = losses_cost + redisp_cost + c_storage

        # compute reward and normalize
        reward = dt_float(-regret/self.max_regret)

        return reward

class L2RPNRewardRegularized(L2RPNReward):
    """
    This reward class extends the original L2RPNReward by normalizing the result
    by dividing it by the number of lines in the grid.
    
    This normalization ensures that the reward scale remains consistent regardless
    of the grid size, making it more suitable for comparing performance across
    different grid topologies or for curriculum learning scenarios.
    
    The original L2RPNReward computes the sum of squared margins for each power line,
    where the margin is defined as (thermal limit - flow) / thermal limit when flow
    is below the thermal limit, and 0 otherwise.
    
    This version normalizes that reward by dividing by the number of lines, so the
    range becomes [0, 1] instead of [0, n_line].
    """
    
    def __init__(self, logger=None):
        super().__init__(logger=logger)
    
    def initialize(self, env):
        # Initialize parent class
        super().initialize(env)
        # Override min/max rewards for normalization
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)  # Instead of n_line in the parent class
        self.n_line = env.backend.n_line
    
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # Get the original reward calculation from parent class
        reward = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
        
        # Normalize the reward by dividing by the number of lines
        if not is_done and not has_error:
            reward = reward / self.n_line
        
        return reward


