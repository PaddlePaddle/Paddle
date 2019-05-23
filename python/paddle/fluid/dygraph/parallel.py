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
import six
import numpy as np

from .. import core
from . import layers
from .. import framework

from ..layers import collective
from . import to_variable

__all__ = ["prepare_context"]

ParallelStrategy = core.ParallelStrategy

__parallel_ctx__clz__ = None


def prepare_context(parallel_strategy):
    global __parallel_ctx__clz__
    assert __parallel_ctx__clz__ is None, "ParallelContext can only be initialized once."
    assert framework.in_dygraph_mode(
    ) is True, "dygraph.parallel.prepare_context should be used with dygrahp mode."
    place = framework._current_expected_place()
    assert place is not None, "dygraph.parallel.prepare_context should be used in fluid.dygraph.guard(place) guard."

    if isinstance(place, core.CUDAPlace):
        __parallel_ctx__clz__ = core.NCCLParallelContext(parallel_strategy,
                                                         place)
    else:
        # TODO(Yancey1989): add Gloo Parallel Context to support CPU parallel computation
        assert ("Only support CUDAPlace for now.")
    __parallel_ctx__clz__.init()


class Env(object):
    def __init__(self):
        self._nranks = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        self._local_rank = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self._dev_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        self._trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS",
                                            "").split(",")
        self._current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT", "")

    @property
    def nranks(self):
        return self._nranks

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def dev_id(self):
        return self._dev_id

    @property
    def current_endpoint(self):
        return self._current_endpoint

    @property
    def trainer_endpoints(self):
        return self._trainer_endpoints


class DataParallel(layers.Layer):
    def __init__(self, layers, strategy):
        super(DataParallel,
              self).__init__(layers.full_name() + "_data_parallel")
        self._layers = layers
        self._strategy = strategy

    def forward(self, *inputs, **kwargs):
        return self._layers(*inputs, **kwargs)

    def scale_loss(self, loss):
        if self._strategy.nranks < 2:
            return loss
        loss_scale = to_variable(
            np.array([self._strategy.nranks]).astype("float32"))
        loss_scale.stop_gradient = True
        loss = loss / loss_scale
        return loss

    def apply_collective_grads(self):
        if self._strategy.nranks < 2:
            return

        for param in self._layers.parameters():
            if param.trainable and param._ivar._grad_ivar():
                g_var = framework.Variable(
                    block=self._helper.main_program.current_block(),
                    name=param._ivar._grad_name(),
                    stop_gradient=True,
                    ivar=param._ivar._grad_ivar())
                collective._allreduce(g_var, g_var, sync_mode=True)
