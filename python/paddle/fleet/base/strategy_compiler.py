#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class StrategyCompilerBase(object):
    def __init__(self):
        pass


class StrategyCompiler(StrategyCompilerBase):
    """
    StrategyCompiler is responsible for meta optimizers combination
    Generally, a user can define serveral distributed strategies that
    can generate serveral meta optimizer. The combination of these 
    meta optimizers should have the right order to apply the optimizers'
    minimize function.
    This class is responsible for the executable distributed optimizer
    generation.
    """
    super(StrategyCompiler, self).__init__()

    def generate_optimizer(self, loss, role_maker, optimizer,
                           userd_defined_strategy, meta_optimizer_list):
        if len(meta_optimizer_list) == 0:
            return optimizer
        else:
            max_idx = 0
            max_len = 0
            candidates = []
            # retrieval optimizer candidates
            for idx, opt in enumerate(meta_optimizer_list):
                local_buffer = [opt]
                for meta in meta_optimizer_list:
                    if opt._can_update(meta):
                        local_buffer.append(meta)
                if len(local_buffer) > max_len:
                    max_idx = idx
                candidates.append(local_buffer)

            # rank optimizer candidates
            for idx, opt in enumerate(candidates[max_idx][:-1]):
                opt._update_inner_optimizer(candidates[max_idx][idx + 1])
            return candidates[max_idx][0]
