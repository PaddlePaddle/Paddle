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

import sys
import unittest

import numpy as np

sys.path.append("..")

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle

paddle.enable_static()


def output_hist(out):
    hist, _ = np.histogram(out, range=(-10, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


class XPUTestRandIntOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'randint'
        self.use_dynamic_create_class = False

    class TestXPURandIntOp(XPUOpTest):
        def setUp(self):
            self.op_type = "randint"
            self.dtype = self.in_type
            self.set_attrs()

            self.atol = 1e-4
            np.random.seed(10)
            self.inputs = {}
            self.outputs = {"Out": np.zeros((10000, 784)).astype("float32")}
            self.attrs = {
                "shape": [10000, 784],
                "low": -10,
                "high": 10,
                "seed": 10,
            }
            self.output_hist = output_hist

        def set_attrs(self):
            pass

        def test_check_output(self):
            self.check_output_customized(self.verify_output)

        def verify_output(self, outs):
            hist, prob = self.output_hist(np.array(outs[0]))
            np.testing.assert_allclose(hist, prob, rtol=0, atol=0.001)


support_types = get_xpu_op_support_types('randint')
for stype in support_types:
    create_test_class(globals(), XPUTestRandIntOp, stype)


if __name__ == "__main__":
    unittest.main()
