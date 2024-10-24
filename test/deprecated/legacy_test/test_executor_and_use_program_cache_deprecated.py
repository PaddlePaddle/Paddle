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

import sys
import unittest

import numpy as np

sys.path.append("../../legacy_test")

import paddle
from paddle import base

paddle.enable_static()


class TestExecutor(unittest.TestCase):
    def test_mul(self):
        main_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(main_program, startup_program):
            a = paddle.static.data(name='a', shape=[-1, 784], dtype='float32')
            b = paddle.static.data(name='b', shape=[784, 100], dtype='float32')
            a.desc.set_need_check_feed(False)
            b.desc.set_need_check_feed(False)
            output = paddle.matmul(x=a, y=b)

        # Compute with numpy
        a_np = np.random.random((100, 784)).astype('float32')
        b_np = np.random.random((784, 100)).astype('float32')
        out_np = np.dot(a_np, b_np)

        place = paddle.CPUPlace()
        exe = base.Executor(place)

        def _train(use_program_cache, max_iters=1):
            import time

            run_time = 0.0
            for i in range(max_iters):
                begin = time.time()
                outs = exe.run(
                    program=main_program,
                    feed={'a': a_np, 'b': b_np},
                    fetch_list=[output],
                    use_program_cache=use_program_cache,
                )
                end = time.time()
                run_time += end - begin
                out = outs[0]
                self.assertEqual((100, 100), out.shape)
                np.testing.assert_allclose(out, out_np, rtol=1e-05)
            return run_time

        max_iters = 3
        run_time_with_cache = _train(
            use_program_cache=True, max_iters=max_iters
        )
        print(f"run time with program cache: {run_time_with_cache:f}")

        run_time_without_cache = _train(
            use_program_cache=False, max_iters=max_iters
        )
        print(f"run time without program cache: {run_time_without_cache:f}")

        run_time_with_cache = _train(
            use_program_cache=True, max_iters=max_iters
        )
        print(f"run time with program cache: {run_time_with_cache:f}")

        run_time_with_cache = _train(
            use_program_cache=True, max_iters=max_iters
        )
        print(f"run time with program cache: {run_time_with_cache:f}")


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
