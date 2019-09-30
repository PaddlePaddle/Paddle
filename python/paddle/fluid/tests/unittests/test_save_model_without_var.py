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

import numpy as np
import warnings
import unittest
import paddle
import paddle.fluid as fluid
from paddle.fluid.layers.device import get_places
from paddle.fluid.executor import as_numpy


class TestSaveModelWithoutVar(unittest.TestCase):
    def test_no_var_save(self):
        data = fluid.layers.data(
            name='data',
            shape=[-1, 1],
            dtype='float32',
            append_batch_size=False)
        data_plus = data + 1

        if fluid.core.is_compiled_with_cuda():
            place = fluid.core.CUDAPlace(0)
        else:
            place = fluid.core.CPUPlace()

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            fluid.io.save_inference_model(
                dirname='test',
                feeded_var_names=['data'],
                target_vars=[data_plus],
                executor=exe,
                model_filename='model',
                params_filename='params')
            expected_warn = "no variable in your model, please ensure there are any variables in your model to save"
            self.assertTrue(len(w) > 0)
            self.assertTrue(expected_warn == str(w[0].message))


if __name__ == '__main__':
    unittest.main()
