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

import os
import unittest

import numpy as np
from eager_op_test import OpTest


class TestDnnlFusedMatMulOp(OpTest):
    def generate_data(self):
        self.x = np.random.random((25, 2, 2)).astype("float32")
        self.y = np.random.random((25, 2, 2)).astype("float32")
        self.alpha = 1.0
        self.out = self.alpha * np.matmul(self.x, self.y)

    def set_attributes(self):
        self.alpha = self.alpha if hasattr(self, 'alpha') else 1.0
        self.attrs = {'alpha': self.alpha}

    def setUp(self):
        # Set max isa, otherwise fails on SKX and earlier
        os.environ["DNNL_MAX_CPU_ISA"] = "AVX"
        self.op_type = "fused_matmul"
        self._cpu_only = True
        self.use_mkldnn = True
        self.generate_data()
        self.set_attributes()
        self.attrs['use_mkldnn'] = True

        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestDnnlFusedMatMulOpInt8(TestDnnlFusedMatMulOp):
    # Due to limitation in int8 matmul implementation
    # on older platforms (BDW, SKX) we needed to reduce
    # range from [-127, 127] to [-63, 63]
    def quantize(self, tensor):
        scale = 63.0 / np.abs(np.amax(tensor))
        quantized = np.round(scale * tensor).astype("int8")
        return scale, quantized

    def generate_data(self):
        x_float = np.random.random((12, 9)).astype("float32")
        self.x_scale, self.x = self.quantize(x_float)

        y_float = np.random.random((9, 12)).astype("float32")
        self.y_scale, self.y = self.quantize(y_float)

        out_float = np.matmul(x_float, y_float)
        self.out_scale, self.out = self.quantize(out_float)

    def set_attributes(self):
        self.attrs = {
            'Scale_x': self.x_scale,
            'Scale_y': self.y_scale,
            'Scale_out': self.out_scale,
        }

    def test_check_output(self):
        self.check_output(atol=1, check_dygraph=False)


class TestDnnlFusedMatMulOpInt8ForceFP32(TestDnnlFusedMatMulOpInt8):
    def generate_data(self):
        x_float = np.random.random((12, 9)).astype("float32")
        self.x_scale, self.x = self.quantize(x_float)

        y_float = np.random.random((9, 12)).astype("float32")
        self.y_scale, self.y = self.quantize(y_float)

        out_float = np.matmul(x_float, y_float)
        self.out = out_float

    def set_attributes(self):
        self.attrs = {
            'Scale_x': self.x_scale,
            'Scale_y': self.y_scale,
            'force_fp32_output': True,
        }


if __name__ == "__main__":
    from paddle import enable_static

    enable_static()
    unittest.main()
