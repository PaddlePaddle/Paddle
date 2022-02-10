#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import unittest
import sys
import paddle
import paddle.fluid as fluid

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestIpuShard(unittest.TestCase):
    def _test(self):
        # build graph
        a = paddle.static.data(name='data', shape=[None, 1], dtype='int32')
        b = a + 2  # scale : scale * x + bias, ipu_index : no

        with paddle.fluid.ipu_shard(ipu_index=1):
            c = b + 1  # scale, ipu_index : 1
            with paddle.fluid.ipu_shard(ipu_index=2):
                d = c * 2  # scale, ipu_index : 2
            with paddle.fluid.ipu_shard(ipu_index=3):
                e = d + 3  # scale, ipu_index : 3
                with paddle.fluid.ipu_shard(ipu_index=1):
                    e = e + 3  # scale, ipu_index : 1
                    with paddle.fluid.ipu_shard(ipu_index=2):
                        e = e + 3  # scale, ipu_index : 2

        with paddle.fluid.ipu_shard(ipu_index=1):
            f = paddle.tensor.pow(e, 2.0)  # pow, ipu_index : 1

        with paddle.fluid.ipu_shard(ipu_index=2):
            g = f - 1  # scale, ipu_index : 2

        h = g + 1  # scale, ipu_index : no

        ipu_index_list = []
        main_prog = paddle.static.default_main_program()
        for op in main_prog.global_block().ops:
            if op.desc.has_attr("ipu_index"):
                ipu_index_list.append(op.desc.attr("ipu_index"))

        return ipu_index_list

    def test_ipu_shard(self):
        ipu_index_list = self._test()
        expected_ipu_index_list = [1, 2, 3, 1, 2, 1, 2]
        self.assertTrue(
            np.allclose(
                ipu_index_list, expected_ipu_index_list, atol=0))


if __name__ == "__main__":
    unittest.main()
