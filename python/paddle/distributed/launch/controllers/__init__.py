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

__all__ = []

from .collective import CollectiveController, CollectiveElasticController
from .ipu_controller import IPUController
from .ps import PSController
from .rpc import RpcController

# the order is extremely important
_controllers = [
    IPUController,
    CollectiveElasticController,
    PSController,
    RpcController,
    CollectiveController,
]


def init(ctx):
    for c in _controllers:
        if c.enable(ctx):
            ctx.print()
            return c(ctx)
