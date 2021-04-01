#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.dygraph.layers import Layer
from .meta_parallel_base import MetaParallelBase
from ..utils.hybrid_parallel_util import *
import paddle


class DataParallel(MetaParallelBase):
    def _prepare_for_model(self):
        pass
        # broadcast_mp_parameters(self._layers, self._hcg)
        #broadcast_dp_parameters(self._layers, self._hcg)

    def __init__(self, layers, **kwargs):
        super(DataParallel, self).__init__(layers, **kwargs)
        self.strategy = kwargs.get('strategy', None)
        assert self.strategy is not None

        self._layers = paddle.DataParallel(
            layers,
            comm_buffer_size=self.strategy.fuse_grad_size_in_MB,
            last_comm_buffer_size=self.strategy.last_comm_group_size_MB)
        self._prepare_for_model()

    def _pre_forward(self, *inputs, **kwargs):
        pass
        # return broadcast_input_data(self._hcg, *inputs, **kwargs)

    def forward(self, *inputs, **kwargs):
        # inputs, kwargs = self._pre_forward(*inputs, **kwargs)

        output = self._layers(*inputs, **kwargs)

        # self._post_forward(output)

        return output

    def _post_forward(self, output):
        pass

    def _pre_backward(self, loss):
        pass

    def backward_impl(self, loss, parameters):
        # self._loss_parameters = parameters
        # self._pre_backward(loss)

        loss.backward()

        # self._post_backward(loss)

    def _post_backward(self, loss):
        pass
        # if not self._loss_parameters:
        #     fused_allreduce_gradients(list(self._layers.parameters()), self._hcg)
        # else:
        #     fused_allreduce_gradients(list(self._loss_parameters), self._hcg)
