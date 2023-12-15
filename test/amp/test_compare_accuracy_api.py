# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "not support cpu TestCompareAccuracyApi"
)
class TestCompareAccuracyApi(unittest.TestCase):
    def calc(self, path, dtype):
        paddle.base.core.set_nan_inf_debug_path(path)
        x = paddle.to_tensor(
            [2000, 3000, 4, 0], place=core.CUDAPlace(0), dtype=dtype
        )
        y = paddle.to_tensor(
            [100, 500, 2, 10000], place=core.CUDAPlace(0), dtype=dtype
        )
        # normal
        z1 = x + y
        # inf
        z2 = x * y

    def test(self):
        paddle.set_flags(
            {"FLAGS_check_nan_inf": 1, "FLAGS_check_nan_inf_level": 3}
        )
        fp32_path = "workerlog_fp32_log_dir"
        fp16_path = "workerlog_fp16_log_dir"
        self.calc(fp32_path, "float32")
        self.calc(fp16_path, "float16")

        out_excel = "compary_accuracy_out_excel.csv"
        paddle.amp.debugging.compare_accuracy(
            fp32_path,
            fp16_path,
            out_excel,
            loss_scale=1,
            dump_all_tensors=False,
        )

    def test2(self):
        fp32_path = "workerlog_fp32_log_dir"
        fp16_path = "workerlog_fp16_null_log_dir"
        self.calc(fp32_path, "float32")
        out_excel = "compary_accuracy_out_excel_2.csv"
        paddle.amp.debugging.compare_accuracy(
            fp32_path,
            fp16_path,
            out_excel,
            loss_scale=1,
            dump_all_tensors=False,
        )


if __name__ == '__main__':
    unittest.main()
