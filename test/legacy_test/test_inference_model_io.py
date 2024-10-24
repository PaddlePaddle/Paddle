#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.base import core, executor
from paddle.distributed.io import (
    load_inference_model_distributed,
)
from paddle.static.io import load_inference_model

paddle.enable_static()


class TestLoadInferenceModelError(unittest.TestCase):

    def test_load_model_not_exist(self):
        place = core.CPUPlace()
        exe = executor.Executor(place)
        self.assertRaises(
            ValueError, load_inference_model, './test_not_exist_dir/model', exe
        )
        self.assertRaises(
            ValueError,
            load_inference_model_distributed,
            './test_not_exist_dir',
            exe,
        )


if __name__ == '__main__':
    unittest.main()
