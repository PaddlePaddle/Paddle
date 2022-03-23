#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.static

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestIpuShard(unittest.TestCase):
    def _test(self):
        # build graph
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            a = paddle.static.data(name='data', shape=[None, 1], dtype='int32')
            b = a + 2  # scale : scale * x + bias, ipu_index : no

            with paddle.static.ipu_shard_guard(index=1):
                c = b + 1  # scale, ipu_index : 1
                with paddle.static.ipu_shard_guard(index=2):
                    d = c * 2  # scale, ipu_index : 2
                with paddle.static.ipu_shard_guard(index=3):
                    e = d + 3  # scale, ipu_index : 3
                    with paddle.static.ipu_shard_guard(index=1):
                        e = e + 3  # scale, ipu_index : 1
                        with paddle.static.ipu_shard_guard(index=2):
                            e = e + 3  # scale, ipu_index : 2

            with paddle.static.ipu_shard_guard(index=1):
                f = paddle.tensor.pow(e, 2.0)  # pow, ipu_index : 1

            with paddle.static.ipu_shard_guard(index=2):
                g = f - 1  # scale, ipu_index : 2

            h = g + 1  # scale, ipu_index : no

        ipu_index_list = []
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


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestIpuPipeline(unittest.TestCase):
    def _test(self):
        # build graph
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            a = paddle.static.data(name='data', shape=[None, 1], dtype='int32')
            b = a + 2  # scale : scale * x + bias, ipu_stage : no

            with paddle.static.ipu_shard_guard(stage=1):
                c = b + 1  # scale, ipu_stage : 1
                with paddle.static.ipu_shard_guard(stage=2):
                    d = c * 2  # scale, ipu_stage : 2
                with paddle.static.ipu_shard_guard(stage=3):
                    e = d + 3  # scale, ipu_stage : 3
                    with paddle.static.ipu_shard_guard(stage=1):
                        e = e + 3  # scale, ipu_stage : 1
                        with paddle.static.ipu_shard_guard(stage=2):
                            e = e + 3  # scale, ipu_stage : 2

            with paddle.static.ipu_shard_guard(stage=1):
                f = paddle.tensor.pow(e, 2.0)  # pow, ipu_stage : 1

            with paddle.static.ipu_shard_guard(stage=2):
                g = f - 1  # scale, ipu_stage : 2

            h = g + 1  # scale, ipu_stage : no

        ipu_index_list = []
        for op in main_prog.global_block().ops:
            if op.desc.has_attr("ipu_stage"):
                ipu_index_list.append(op.desc.attr("ipu_stage"))

        return ipu_index_list

    def test_ipu_shard(self):
        ipu_index_list = self._test()
        expected_ipu_index_list = [1, 2, 3, 1, 2, 1, 2]

        self.assertTrue(
            np.allclose(
                ipu_index_list, expected_ipu_index_list, atol=0))


if __name__ == "__main__":
    unittest.main()
