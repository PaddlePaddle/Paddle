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
from . import parallel_helper
from .. import framework
from ..layers import collective
from . import to_variable

__all__ = ["prepare_context"]

ParallelStrategy = core.ParallelStrategy


def prepare_context(strategy=None):
    if strategy is None:
        strategy = ParallelStrategy()
        strategy.nranks = Env().nranks
        strategy.local_rank = Env().local_rank
        strategy.trainer_endpoints = Env().trainer_endpoints
        strategy.current_endpoint = Env().current_endpoint
    if strategy.nranks < 2:
        return
    assert framework.in_dygraph_mode() is True,\
        "dygraph.parallel.prepare_context should be used with dygrahp mode."
    place = framework._current_expected_place()
    assert place is not None, \
        "dygraph.parallel.prepare_context should be used in fluid.dygraph.guard(place) guard."
    if isinstance(place, core.CUDAPlace):
        parallel_helper._set_parallel_ctx(
            core.NCCLParallelContext(strategy, place))
    else:
        # TODO(Yancey1989): add Gloo Parallel Context to support CPU parallel computation
        assert ("Only support CUDAPlace for now.")
    parallel_helper._init_parallel_ctx()
    return strategy


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
    """
    Runs the module with data parallelism.

    Currently, DataParallel only supports to run the dynamic graph
    with multi-process. The usage is:
    `python -m paddle.distributed.launch --gpus 2 dynamic_graph_test.py`.
    And the content of `dynamic_graph_test.py` is the code of examples.

    Examples:
        .. code-block:: python

           import numpy as np
           import paddle.fluid as fluid
           import paddle.fluid.dygraph as dygraph
           from paddle.fluid.optimizer import AdamOptimizer
           from paddle.fluid.dygraph.nn import FC
           from paddle.fluid.dygraph.base import to_variable

           place = fluid.CUDAPlace(0)
           with fluid.dygraph.guard(place=place):

               # prepare the data parallel context
               strategy=dygraph.parallel.prepare_context()

               fc_layer = FC("FC", 10, act="softmax")
               adam = fluid.optimizer.AdamOptimizer()

               # make the module become the data parallelism module
               fc_layer = dygraph.parallel.DataParallel(fc_layer, strategy)

               x_data = np.random.random(size=[10, 1]).astype(np.float32)
               data = to_variable(x_data)

               hidden = fc_layer(data)
               avg_loss = fluid.layers.mean(hidden)

               # scale the loss according to the number of trainers.
               avg_loss = fc_layer.scale_loss(avg_loss)

               avg_loss.backward()

               # collect the gradients of trainers.
               fc_layer.apply_collective_grads()

               adam.minimize(avg_loss)
               fc_layer.clear_gradients()

    Args:
        layers(Layer): The module that should be executed by data parallel.
        strategy(ParallelStrategy): The strategy of data parallelism.

    Returns:
        Layer: The data paralleled module.
    """

    def __init__(self, layers, strategy):
        super(DataParallel,
              self).__init__(layers.full_name() + "_data_parallel")

        self._layers = layers
        self._strategy = strategy

    def forward(self, *inputs, **kwargs):
        return self._layers(*inputs, **kwargs)

    def scale_loss(self, loss):
        """
        Scale the loss. In data parallel mode, the loss should be scale with
        the number of trainers. If not in data parallel mode, return the loss
        directly.

        Args:
            loss(Layer): The loss of the current Model.

        Returns:
            Layer: the scaled loss.
        """
        if not self._is_data_parallel_mode():
            return loss

        loss_scale = to_variable(
            np.array([self._strategy.nranks]).astype("float32"))
        loss_scale.stop_gradient = True
        loss = loss / loss_scale
        return loss

    def apply_collective_grads(self):
        """
        AllReduce the Parameters' gradient.
        """
        if not self._is_data_parallel_mode():
            return

        for param in self._layers.parameters():
            # NOTE(zcd): The grad_ivar maybe no generated.
            if param.trainable and param._ivar._grad_ivar():
                g_var = framework.Variable(
                    block=self._helper.main_program.current_block(),
                    name=param._ivar._grad_name(),
                    stop_gradient=True,
                    ivar=param._ivar._grad_ivar())
                collective._allreduce(g_var, g_var, sync_mode=True)

    def _is_data_parallel_mode(self):
        return self._strategy.nranks > 1
