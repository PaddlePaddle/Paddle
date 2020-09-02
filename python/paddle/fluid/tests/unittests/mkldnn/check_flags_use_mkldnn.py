# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import unicode_literals
from __future__ import print_function

import os
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper

np.random.seed(0)


def check():
    a_np = np.random.uniform(-2, 2, (10, 20, 30)).astype(np.float32)
    helper = LayerHelper(fluid.unique_name.generate(str("test")), act="relu")
    func = helper.append_activation
    with fluid.dygraph.guard():
        a = fluid.dygraph.to_variable(a_np)
        res1 = func(a)
        res2 = fluid.layers.relu(a)
    assert (np.array_equal(res1.numpy(), res2.numpy()))


if __name__ == '__main__':
    try:
        check()
    except Exception as e:
        print(e)
        print(type(e))
