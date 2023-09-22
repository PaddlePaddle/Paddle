#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import paddle


def ref_logaddexp_old(x, y):
    y = np.broadcast_to(y, x.shape)
    out = np.log1p(np.exp(-np.absolute(x - y))) + np.maximum(x, y)
    return out


def ref_logaddexp(x, y):
    return np.logaddexp(x, y)


class TestLogsumexpAPI(unittest.TestCase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.base.core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def api_case(self):
        self.x = np.random.uniform(-1, 1, self.xshape).astype(self.dtype)
        self.y = np.random.uniform(-1, 1, self.yshape).astype(self.dtype)
        out_ref = ref_logaddexp(self.x, self.y)

        # paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.logaddexp(x, y)
        np.testing.assert_allclose(out.numpy(), out_ref, atol=1e-06)

    def test_api(self):
        self.xshape = [1, 2, 3, 4]
        self.yshape = [1, 2, 3, 4]
        self.dtype = np.float64
        self.api_case()

    def test_api_broadcast(self):
        self.xshape = [1, 2, 3, 4]
        self.yshape = [1, 2, 3, 1]
        self.dtype = np.float32
        self.api_case()

    def test_api_bigdata(self):
        self.xshape = [10, 200, 300]
        self.yshape = [10, 200, 300]
        self.dtype = np.float32
        self.api_case()

    def test_api_int32(self):
        self.xshape = [10, 200, 300]
        self.yshape = [10, 200, 300]
        self.dtype = np.int32
        self.api_case()

    def test_api_int64(self):
        self.xshape = [10, 200, 300]
        self.yshape = [10, 200, 300]
        self.dtype = np.int64
        self.api_case()


if __name__ == '__main__':
    unittest.main()
