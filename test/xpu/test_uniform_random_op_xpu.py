#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()
from paddle.base import core

typeid_dict = {
    'int32': int(core.VarDesc.VarType.INT32),
    'int64': int(core.VarDesc.VarType.INT64),
    'float32': int(core.VarDesc.VarType.FP32),
    'float16': int(core.VarDesc.VarType.FP16),
    'bfloat16': int(core.VarDesc.VarType.BF16),
    'bool': int(core.VarDesc.VarType.BOOL),
    'int8': int(core.VarDesc.VarType.INT8),
    'uint8': int(core.VarDesc.VarType.UINT8),
    'float64': int(core.VarDesc.VarType.FP64),
}


def output_hist(out):
    if out.dtype == np.uint16:
        out = convert_uint16_to_float(out)
    hist, _ = np.histogram(out, range=(-5, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones(10)
    return hist, prob


from op_test import convert_uint16_to_float


class XPUTestUniformRandomOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'uniform_random'
        self.use_dynamic_create_class = False

    class TestUniformRandomOp(XPUOpTest):
        def init(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = "uniform_random"
            self.python_api = paddle.uniform

        def setUp(self):
            self.init()
            self.inputs = {}
            self.use_mkldnn = False
            self.set_attrs()
            paddle.seed(10)

            self.outputs = {"Out": np.zeros((1000, 784), dtype=self.dtype)}

        def set_attrs(self):
            self.attrs = {
                "shape": [1000, 784],
                "min": -5.0,
                "max": 10.0,
                "dtype": typeid_dict[self.in_type_str],
            }
            self.output_hist = output_hist

        def test_check_output(self):
            self.check_output_with_place_customized(
                self.verify_output, self.place
            )

        def verify_output(self, outs):
            hist, prob = self.output_hist(np.array(outs[0]))
            np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)

    class TestMaxMinAreInt(TestUniformRandomOp):
        def set_attrs(self):
            self.attrs = {
                "shape": [1000, 784],
                "min": -5,
                "max": 10,
                "dtype": typeid_dict[self.in_type_str],
            }
            self.output_hist = output_hist


support_types = get_xpu_op_support_types('uniform_random')
for stype in support_types:
    create_test_class(globals(), XPUTestUniformRandomOp, stype)

if __name__ == "__main__":
    unittest.main()
