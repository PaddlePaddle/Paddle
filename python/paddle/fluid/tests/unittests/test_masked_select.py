# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import sys
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.executor import Executor


class TestMaskedSelect(unittest.TestCase):
    def test_masked_select(self):

        mask_shape = [4, 1]
        shape = [4, 4]
        data = np.random.random(mask_shape).astype("float32")
        input_data = np.random.random(shape).astype("float32")
        mask_data = data > 0.5
        mask_data_b = np.broadcast_to(mask_data, shape)
        npresult = input_data[np.where(mask_data_b)]

        input_var = layers.create_tensor(dtype="float32", name="input")
        mask_var = layers.create_tensor(dtype="bool", name="mask")

        output = layers.masked_select(input=input_var, mask=mask_var)
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = Executor(place)
            result = exe.run(fluid.default_main_program(),
                             feed={"input": input_data,
                                   "mask": mask_data},
                             fetch_list=[output])

            self.assertTrue(np.isclose(np_result, result).all())


if __name__ == "__main__":
    unittest.main()
