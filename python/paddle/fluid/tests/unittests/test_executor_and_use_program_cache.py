#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest

import numpy
import paddle.fluid.core as core
from paddle.fluid.executor import Executor
from paddle.fluid.layers import mul, data


class TestExecutor(unittest.TestCase):
    def test_mul(self):
        a = data(name='a', shape=[784], dtype='float32')
        b = data(
            name='b',
            shape=[784, 100],
            dtype='float32',
            append_batch_size=False)
        output = mul(x=a, y=b)
        place = core.CPUPlace()
        a_np = numpy.random.random((100, 784)).astype('float32')
        b_np = numpy.random.random((784, 100)).astype('float32')
        exe = Executor(place)
        import time
        use_cache = True
        step_num = 3
        run_time = 0.0
        for i in range(step_num):
            begin = time.time()
            outs = exe.run(feed={'a': a_np,
                                 'b': b_np},
                           fetch_list=[output.name],
                           use_program_cache=use_cache)
            end = time.time()
            run_time += end - begin
            out = outs[0]
            self.assertEqual((100, 100), out.shape)
            self.assertTrue(numpy.allclose(out, numpy.dot(a_np, b_np)))
        print("run time %f" % run_time)
        use_cache = False
        run_time = 0.0
        for i in range(step_num):
            begin = time.time()
            outs = exe.run(feed={'a': a_np,
                                 'b': b_np},
                           fetch_list=[output.name],
                           use_program_cache=use_cache)
            end = time.time()
            run_time += end - begin
            out = outs[0]
            self.assertEqual((100, 100), out.shape)
            self.assertTrue(numpy.allclose(out, numpy.dot(a_np, b_np)))
        print("run time %f" % run_time)
        use_cache = True
        run_time = 0.0
        for i in range(step_num):
            begin = time.time()
            outs = exe.run(feed={'a': a_np,
                                 'b': b_np},
                           fetch_list=[output.name],
                           use_program_cache=use_cache)
            end = time.time()
            run_time += end - begin
            out = outs[0]
            self.assertEqual((100, 100), out.shape)
            self.assertTrue(numpy.allclose(out, numpy.dot(a_np, b_np)))
        print("run time %f" % run_time)

        use_cache = True
        run_time = 0.0
        for i in range(step_num):
            begin = time.time()
            outs = exe.run(feed={'a': a_np,
                                 'b': b_np},
                           fetch_list=[output],
                           use_program_cache=use_cache)
            end = time.time()
            run_time += end - begin
            out = outs[0]
            self.assertEqual((100, 100), out.shape)
            self.assertTrue(numpy.allclose(out, numpy.dot(a_np, b_np)))
        print("run time %f" % run_time)


if __name__ == '__main__':
    unittest.main()
