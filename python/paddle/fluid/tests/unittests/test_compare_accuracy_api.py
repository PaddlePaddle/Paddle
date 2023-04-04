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

import os
import unittest

import paddle
from paddle.fluid import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestCompareAccuracyApi(unittest.TestCase):
    def test_num_nan_inf(self):
        path1 = "workerlog_fp32_log_dir"
        paddle.fluid.core.set_nan_inf_debug_path(path1)

        paddle.set_flags(
            {"FLAGS_check_nan_inf": 1, "FLAGS_check_nan_inf_level": 3}
        )
        x = paddle.to_tensor(
            [2, 3, 4, 0], place=core.CUDAPlace(0), dtype='float32'
        )
        y = paddle.to_tensor(
            [1, 5, 2, 0], place=core.CUDAPlace(0), dtype='float32'
        )
        z = paddle.add(x, y)

        path2 = "workerlog_fp16_log_dir"
        paddle.fluid.core.set_nan_inf_debug_path(path2)

        paddle.set_flags(
            {"FLAGS_check_nan_inf": 1, "FLAGS_check_nan_inf_level": 3}
        )
        x = paddle.to_tensor(
            [2, 3, 4, 0], place=core.CUDAPlace(0), dtype='float16'
        )
        y = paddle.to_tensor(
            [1, 5, 2, 0], place=core.CUDAPlace(0), dtype='float16'
        )
        z = paddle.add(x, y)

        out_excel = "compary_accuracy_out_excel.csv"
        print(os.path.abspath(out_excel))
        paddle.amp.debugging.compare_accuracy(
            path1, path2, out_excel, loss_scale=1, dump_all_tensors=False
        )


if __name__ == '__main__':
    unittest.main()
