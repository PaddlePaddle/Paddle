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
from paddle.nn import Layer
from paddle.static import InputSpec
from paddle.jit import to_static
import sys


class AbsNet(paddle.nn.Layer):

    def __init__(self):
        super(AbsNet, self).__init__()

    def forward(self, x):
        x = paddle.abs(x)
        return x


if __name__ == '__main__':
    # build network
    model = AbsNet()
    # save inferencing format model
    net = to_static(model,
                    input_spec=[InputSpec(shape=[None, 1, 28, 28], name='x')])
    paddle.jit.save(net, sys.argv[1])
