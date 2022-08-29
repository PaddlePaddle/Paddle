# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed import fleet


def init_parallel_env(mode, global_batch_size, seed=1024):
    '''
        Args:
            mode:(str) DP1-MP1-PP1-SH1-O1
    '''

    def parse_mode(mode):
        assert "DP" == mode[:2]
        assert "-MP" in mode
        assert "-PP" in mode
        assert "-SH" in mode
        assert "-O" in mode
        modes = mode.split("-")
        DP = int(modes[0][2:])
        MP = int(modes[1][2:])
        PP = int(modes[2][2:])
        SH = int(modes[3][2:])
        Ostage = int(modes[4][1:])
        return DP, MP, PP, SH, Ostage

    DP, MP, PP, SH, Ostage = parse_mode(mode)

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": DP,
        "mp_degree": MP,
        "pp_degree": PP,
        "sharding_degree": SH
    }

    accumulate_steps = 1

    if PP > 1:
        strategy.pipeline_configs = {
            "accumulate_steps": accumulate_steps,
            "micro_batch_size": global_batch_size // DP // accumulate_steps
        }

    # set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
    fleet.init(is_collective=True, strategy=strategy)

    return fleet.get_hybrid_communicate_group()
