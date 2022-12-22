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

import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F


class TestMemoryReuseExcludeFeedVar(unittest.TestCase):
    def setUp(self):
        self.image_shape = [28, 28]
        self.iteration = 10

    def main_impl(self, place):
        image = fluid.layers.data(
            name='image', shape=self.image_shape, dtype='float32'
        )
        relu_image = F.relu(image)
        loss = paddle.mean(relu_image)

        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        compiled_prog = fluid.CompiledProgram(
            fluid.default_main_program()
        ).with_data_parallel(loss_name=loss.name, build_strategy=build_strategy)

        image_tensor = fluid.LoDTensor()
        np_image = np.random.uniform(
            low=-10, high=10, size=self.image_shape
        ).astype('float32')
        image_tensor.set(np_image, place)

        feed_dict = [{image.name: image_tensor}]

        for _ in range(self.iteration):
            exe.run(compiled_prog, feed=feed_dict, fetch_list=[loss.name])
            np.testing.assert_array_equal(np.array(image_tensor), np_image)

    def test_main(self):
        places = [fluid.CPUPlace()]
        if fluid.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for p in places:
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                with fluid.unique_name.guard():
                    with fluid.scope_guard(fluid.Scope()):
                        self.main_impl(p)


if __name__ == '__main__':
    unittest.main()
