# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import base


class TestMemoryReuseExcludeFeedVar(unittest.TestCase):
    def setUp(self):
        self.image_shape = [28, 28]
        self.iteration = 10

    def main_impl(self, place):
        image = paddle.static.data(
            name='image', shape=[-1, *self.image_shape], dtype='float32'
        )
        relu_image = F.relu(image)
        loss = paddle.mean(relu_image)

        build_strategy = base.BuildStrategy()
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True

        exe = base.Executor(place)
        exe.run(base.default_startup_program())

        compiled_prog = base.CompiledProgram(
            base.default_main_program(), build_strategy=build_strategy
        )

        image_tensor = base.DenseTensor()
        np_image = np.random.uniform(
            low=-10, high=10, size=self.image_shape
        ).astype('float32')
        image_tensor.set(np_image, place)

        feed_dict = [{image.name: image_tensor}]

        for _ in range(self.iteration):
            exe.run(compiled_prog, feed=feed_dict, fetch_list=[loss])
            np.testing.assert_array_equal(np.array(image_tensor), np_image)

    def test_main(self):
        places = [base.CPUPlace()]
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if base.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))

        for p in places:
            with base.program_guard(base.Program(), base.Program()):
                with base.unique_name.guard():
                    with base.scope_guard(base.Scope()):
                        with paddle.pir_utils.OldIrGuard():  # if you need to test in pir mode ,delete this line
                            self.main_impl(p)


if __name__ == '__main__':
    unittest.main()
