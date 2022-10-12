#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.static as static

paddle.enable_static()


class TestSeedOpFixSeed(OpTest):

    def setUp(self):
        self.op_type = "seed"
        self.inputs = {}
        self.attrs = {"seed": 123}
        self.outputs = {"Out": np.asarray((123)).astype('int')}

    def test_check_output(self):
        self.check_output()


class TestSeedOpDiffSeed(OpTest):

    def setUp(self):
        self.op_type = "seed"
        self.inputs = {}
        self.attrs = {"seed": 0}
        self.outputs = {"Out": np.asarray((123)).astype('int')}

    def test_check_output(self):
        self.check_output(no_check_set=["Out"])


class TestDropoutWithRandomSeedGenerator(unittest.TestCase):

    def setUp(self):
        paddle.framework.random.set_random_seed_generator('seed0', 123)
        paddle.framework.random.set_random_seed_generator('seed1', 123)
        self.rng0 = paddle.framework.random.get_random_seed_generator('seed0')
        self.rng1 = paddle.framework.random.get_random_seed_generator('seed1')
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        import paddle.distributed.fleet.meta_parallel.parallel_layers.random as random
        with static.program_guard(static.Program(), static.Program()):
            res1 = random.determinate_seed('seed0')

            exe = static.Executor(place)
            res_list = [res1]
            for i in range(2):
                out1, = exe.run(static.default_main_program(),
                                fetch_list=res_list)
                self.assertEqual(out1, np.cast['int32'](self.rng1.random()))

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)


if __name__ == '__main__':
    unittest.main()
