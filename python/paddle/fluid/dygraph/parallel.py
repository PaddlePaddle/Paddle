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

from .. import core
from . import layers
from ..framework import _dygraph_tracer, Variable, _current_expected_place, Parameter
from ..layers import collective

__all__ = ["prepare_context"]

ParallelStrategy = core.ParallelStrategy

__parallel_ctx__clz__ = None


def prepare_context(parallel_strategy, place=None):
    global __parallel_ctx__clz__
    assert __parallel_ctx__clz__ is None, "ParallelContext can only be initialized once."
    if not place:
        place = _current_expected_place()

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
    def __init__(self, layers):
        super(DataParallel,
              self).__init__(layers.full_name() + "_data_parallel")
        self._layers = layers

    def build_once(self, *inputs, **kwargs):
        #TODO(Yancey1989): broadcast all the paramters
        pass

    def forward(self, *inputs, **kwargs):
        def _collective_hook(iop):
            op = _dygraph_tracer()._ops[iop._trace_id]
            for k, v in six.iteritems(op.inputs):
                for ivar in v:
                    g = ivar._grad_ivar()
                    g_var = Variable(
                        block=self._helper.main_program.current_block(),
                        name=ivar._grad_name(),
                        stop_gradient=True,
                        ivar=g)
                    collective._allreduce(g_var, g_var, sync_mode=True)

        outs = self._layers(*inputs, **kwargs)
        for _, op in six.iteritems(_dygraph_tracer()._ops):
            # hook collective ops
            op.iop.register_backward_hooks(_collective_hook, front=True)
        return outs
