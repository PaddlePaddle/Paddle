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

import paddle


class DistributedMode:
    SYNC = 0
    ASYNC = 1
    HALF_ASYNC = 2
    GEO = 3


def get_dist_env(self):
    trainer_id = int(os.getenv('PADDLE_TRAINER_ID', '0'))
    trainer_endpoints = ''
    current_endpoint = ''
    num_trainers = 0
    if os.getenv('PADDLE_TRAINER_ENDPOINTS'):
        trainer_endpoints = os.getenv('PADDLE_TRAINER_ENDPOINTS')
        current_endpoint = trainer_endpoints.split(',')[trainer_id]
        num_trainers = len(trainer_endpoints.split(','))

    return {
        'trainer_id': trainer_id,
        'num_trainers': num_trainers,
        'current_endpoint': current_endpoint,
        'trainer_endpoints': trainer_endpoints
    }


def get_distributed_strategy(dist_strategy):
    from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory

    k_steps = dist_strategy.a_sync_configs["k_steps"]
    strategy = None

    if not dist_strategy.a_sync and k_steps == 0:
        strategy = StrategyFactory.create_sync_strategy()

    if dist_strategy.a_sync and k_steps == 0:
        strategy = StrategyFactory.create_async_strategy()

    if dist_strategy.a_sync and k_steps > 0:
        strategy = StrategyFactory.create_geo_strategy(k_steps)

    if not strategy:
        raise ValueError("k_steps must be invalid value, please check")

    return strategy
