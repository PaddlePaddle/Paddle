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
import paddle.fluid as fluid
from test_eager_deletion_padding_rnn import PaddingRNNTestBase


class TestExecutor(unittest.TestCase):
    def test_mul(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            a = fluid.layers.data(name='a', shape=[784], dtype='float32')
            b = fluid.layers.data(
                name='b',
                shape=[784, 100],
                dtype='float32',
                append_batch_size=False)
            output = fluid.layers.mul(x=a, y=b)

        # Compute with numpy
        a_np = numpy.random.random((100, 784)).astype('float32')
        b_np = numpy.random.random((784, 100)).astype('float32')
        out_np = numpy.dot(a_np, b_np)

        place = core.CPUPlace()
        exe = fluid.Executor(place)

        def _train(use_program_cache, max_iters=1):
            import time

            run_time = 0.0
            for i in range(max_iters):
                begin = time.time()
                outs = exe.run(program=main_program,
                               feed={'a': a_np,
                                     'b': b_np},
                               fetch_list=[output.name],
                               use_program_cache=use_program_cache)
                end = time.time()
                run_time += end - begin
                out = outs[0]
                self.assertEqual((100, 100), out.shape)
                self.assertTrue(numpy.allclose(out, out_np))
            return run_time

        max_iters = 3
        run_time_with_cache = _train(
            use_program_cache=True, max_iters=max_iters)
        print("run time with program cache: %f" % run_time_with_cache)

        run_time_without_cache = _train(
            use_program_cache=False, max_iters=max_iters)
        print("run time without program cache: %f" % run_time_without_cache)

        run_time_with_cache = _train(
            use_program_cache=True, max_iters=max_iters)
        print("run time with program cache: %f" % run_time_with_cache)
        self.assertLess(run_time_with_cache, run_time_without_cache)

        run_time_with_cache = _train(
            use_program_cache=True, max_iters=max_iters)
        print("run time with program cache: %f" % run_time_with_cache)
        self.assertLess(run_time_with_cache, run_time_without_cache)


class ExecutorPaddingRNNTest(PaddingRNNTestBase):
    def test_executor_use_program_cache(self):
        # Set parallel to False to use the default executor.
        self.compare_padding_static_mode(parallel=False, use_program_cache=True)


if __name__ == '__main__':
    unittest.main()
