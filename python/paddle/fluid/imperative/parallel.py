# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except jin compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from .. import core
from paddle.fluid import framework

__all__ = ["prepare_context"]

ParallelStrategy = core.ParallelStrategy

__init__clz__ = None


def prepare_context(parallel_strategy, place):
    if isinstance(place, core.CUDAPlace):
        __init__clz__ = core.NCCLParallelContext(parallel_strategy, place)
    else:
        # TODO(Yancey1989): add Gloo Parallel Context to support CPU parallel training
        assert ("Only support CUDAPlace")
    __init__clz__.init()


def nranks():
    return int(os.getenv("PADDLE_TRAINERS_NUM", "1"))


def local_rank():
    return int(os.getenv("PADDLE_TRAINER_ID", "0"))


def dev_id():
    return int(os.getenv("FLAGS_selected_gpus", "0"))


def trainer_endpoints():
    return os.getenv("PADDLE_TRAINER_ENDPOINTS", "")


def current_endpoint():
    return os.getenv("PADDLE_CURRENT_ENDPOINT", "")
