#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
from op_test import OpTest


class TestUniformRandomBatchSizeLike(OpTest):
    def setUp(self):
        self.op_type = "uniform_random_batch_size_like"
        self.inputs = {'Input': np.zeros((500, 2000), dtype="float32")}
        self.attrs = {'min': 1., 'max': 2., 'shape': [-1, 2000]}
        self.outputs = {'Out': np.zeros((500, 2000), dtype='float32')}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        self.assertEqual(outs[0].shape, (500, 2000))
        hist, _ = np.histogram(outs[0], range=(1, 2))
        hist = hist.astype("float32")
        hist /= float(outs[0].size)
        prob = 0.1 * np.ones((10))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.01), "hist: " + str(hist))


if __name__ == "__main__":
    unittest.main()
