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

import numpy as np
import paddle.fluid as fluid

import unittest


class TestApi(unittest.TestCase):
    def test_grnn_api(self):
        x = fluid.data(name='x', shape=[10, 16], dtype='float32')
        param_in = fluid.ParamAttr(name="param_in")
        param_hidden = fluid.ParamAttr(name="param_hidden")
        y = fluid.layers.search_grnn(x, 16, 16, param_attr_in, param_hidden)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        out = exe.run(fluid.default_main_program(),
                      feed={'x': np.random.rand([10, 16]).astype('float32')},
                      fetch=[y])
        print(out[0].numpy())


if __name__ == '__main__':
    unittest.main()
