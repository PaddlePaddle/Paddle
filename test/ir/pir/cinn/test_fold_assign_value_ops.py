# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy

import paddle


class TestFoldAssignValueOps(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_result(self, dy_compute):
        dy_out = dy_compute()
        st_out = paddle.jit.to_static(full_graph=True, backend="CINN")(
            dy_compute
        )()
        numpy.testing.assert_allclose(dy_out, st_out, atol=1e-6, rtol=1e-6)

    def test_eval(self):
        def func():
            x = paddle.full(shape=[2, 2], fill_value=0, dtype="int64")
            values = numpy.array([1, 2, 3, 4], dtype=numpy.int64)
            o = paddle.assign(values, x)
            o = paddle.cast(o, dtype="int32")
            return o

        self.compare_result(func)


if __name__ == "__main__":
    unittest.main()
