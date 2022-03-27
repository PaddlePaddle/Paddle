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
"""
A simple elementwise add net for infrt benchmark. 
"""

import paddle
from paddle.jit import to_static


class SimpleNet(paddle.nn.Layer):
    """
    A simple elementwise add net for infrt benchmark. 
    """

    def __init__(self, layers, num):
        super(SimpleNet, self).__init__()
        self.num = num
        self.layers = layers
        self.y = self.create_parameter(shape=[self.num, self.num])

    @to_static
    def forward(self, x):
        out = paddle.add(x, self.y)
        for number in range(self.layers - 1):
            out = paddle.add(x, out)
        return out

    def save(self):
        self.eval()
        x = paddle.rand([self.num, self.num])
        out = self(x)
        paddle.jit.save(self, "./elt_add/l{}n{}".format(self.layers, self.num))


if __name__ == "__main__":
    for layers in [1, 10, 50]:
        for num in [1, 100, 1000, 10000]:
            print("layers = {}, num = {}".format(layers, num))
            SimpleNet(layers, num).save()
