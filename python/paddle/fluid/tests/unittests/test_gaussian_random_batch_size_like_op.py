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


class TestGaussianRandomBatchSizeLike(OpTest):
    def setUp(self):
        self.op_type = "gaussian_random_batch_size_like"
        self.inputs = {'Input': np.zeros((500, 2000), dtype="float32")}
        self.attrs = {'mean': 1., 'std': 2., 'shape': [-1, 2000]}
        self.outputs = {'Out': np.zeros((500, 2000), dtype='float32')}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        self.assertEqual(outs[0].shape, (500, 2000))
        hist, _ = np.histogram(outs[0], range=(-3, 5))
        hist = hist.astype("float32")
        hist /= float(outs[0].size)
        data = np.random.normal(size=(500, 2000), loc=1, scale=2)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype("float32")
        hist2 /= float(outs[0].size)
        self.assertTrue(
            np.allclose(
                hist, hist2, rtol=0, atol=0.01),
            "hist: " + str(hist) + " hist2: " + str(hist2))


if __name__ == "__main__":
    unittest.main()
