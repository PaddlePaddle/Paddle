#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

from paddle.fluid.incubate.fleet.parameter_server import version
from paddle.fluid.incubate.fleet.parameter_server.distributed_strategy import TrainerRuntimeConfig
from paddle.fluid.incubate.fleet.parameter_server.distributed_strategy import StrategyFactory

__all__ = ['TrainerRuntimeConfig', 'StrategyFactory', 'fleet']

fleet = None

if version.is_transpiler():
    from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet as fleet_transpiler
    fleet = fleet_transpiler
else:
    from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet as fleet_pslib
    fleet = fleet_pslib
