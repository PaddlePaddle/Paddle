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
import logging


class MetaParallelBase(Layer):
    def __init__(self, layers, **kwargs):
        super(MetaParallelBase,
              self).__init__(layers.full_name() + "_meta_parallel_base")
        self._layers = layers
        self.strategy = kwargs.get('strategy', None)
        assert self.strategy is not None

        self.use_amp = self.strategy.amp

    def prepare_for_model(self):
        pass

    def _pre_forward(self, *inputs, **kwargs):
        pass

    def forward(self, *inputs, **kwargs):
        self._pre_forward(*inputs, **kwargs)

        output = self._layers(*inputs, **kwargs)

        self._post_forward(output)

        return output

    def _post_forward(self, output):
        pass

    def backward_impl(self, loss, parameters=None):
        self._pre_backward(loss)

        loss.backward()

        self._post_backward(loss)

    def _pre_backward(self, loss):
        pass

    def _post_backward(self, loss):
        pass

    def __call__(self, *inputs, **kwargs):
        if self.use_amp:
            print("start running amp.")
            # logging.info("start running amp.")
            with paddle.amp.auto_cast():
                return super(MetaParallelBase, self).__call__(*inputs, **kwargs)
        else:
            print("not use amp")
            return super(MetaParallelBase, self).__call__(*inputs, **kwargs)
