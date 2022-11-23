# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from hybrid_parallel_pp_alexnet import TestDistPPTraning


class TestPPClipGrad(TestDistPPTraning):

    def build_optimizer(self, model):
        grad_clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2],
                                                       values=[0.001, 0.002],
                                                       verbose=True)
        optimizer = paddle.optimizer.SGD(learning_rate=scheduler,
                                         grad_clip=grad_clip,
                                         parameters=model.parameters())
        return scheduler, optimizer


class TestPPClipGradParamGroup(TestDistPPTraning):

    def build_optimizer(self, model):
        grad_clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2],
                                                       values=[0.001, 0.002],
                                                       verbose=True)
        optimizer = paddle.optimizer.Momentum(learning_rate=scheduler,
                                              grad_clip=grad_clip,
                                              parameters=[{
                                                  "params":
                                                  model.parameters()
                                              }])
        return scheduler, optimizer


if __name__ == "__main__":
    unittest.main()
