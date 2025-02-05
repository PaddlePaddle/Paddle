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

import unittest
import warnings

import paddle
from paddle import base


class TestSaveModelWithoutVar(unittest.TestCase):

    def test_no_var_save(self):
        data = paddle.static.data(name='data', shape=[-1, 1], dtype='float32')
        data_plus = data + 1

        if base.core.is_compiled_with_cuda():
            place = base.core.CUDAPlace(0)
        else:
            place = base.core.CPUPlace()

        exe = base.Executor(place)
        exe.run(base.default_startup_program())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            paddle.static.io.save_inference_model(
                'test',
                data,
                [data_plus],
                exe,
            )
            expected_warn = "no variable in your model, please ensure there are any variables in your model to save"
            self.assertTrue(len(w) > 0)
            self.assertTrue(expected_warn == str(w[-1].message))


if __name__ == '__main__':
    unittest.main()
