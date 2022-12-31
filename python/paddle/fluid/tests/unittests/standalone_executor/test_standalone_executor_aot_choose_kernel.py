# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.framework import set_flags

paddle.enable_static()


def build_resnet50():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        image = paddle.static.data(
            name='image', shape=[32, 3, 224, 224], dtype='float32'
        )
        label = paddle.static.data(name='label', shape=[32], dtype='int64')
        model = paddle.vision.models.resnet50()
        prediction = model(image)
        loss = paddle.nn.functional.cross_entropy(input=prediction, label=label)
        loss = paddle.mean(loss)
        adam = paddle.optimizer.Adam(learning_rate=0.001)
        adam.minimize(loss)

    return main_program, startup_program, loss


class TestAOTChooseKernel(unittest.TestCase):
    def test_aot_choose_kernel(self):
        if not paddle.fluid.core.is_compiled_with_cuda():
            return

        def run(aot_choose_kernel=None):
            paddle.seed(2022)
            np.random.seed(2022)

            main_program, startup_program, loss = build_resnet50()

            scope = paddle.static.Scope()
            exe = paddle.static.Executor()

            set_flags({'FLAGS_cudnn_deterministic': 1})
            if aot_choose_kernel:
                set_flags({'FLAGS_new_executor_static_build': 1})
            else:
                set_flags({'FLAGS_new_executor_static_build': 0})

            with paddle.static.scope_guard(scope):
                exe.run(startup_program)

                for i in range(10):
                    feed = {
                        'image': np.random.randint(
                            0, 256, size=[32, 3, 224, 224]
                        ).astype('float32'),
                        'label': np.random.randint(0, 1000, size=[32]).astype(
                            'int64'
                        ),
                    }
                    loss_ = exe.run(main_program, feed=feed, fetch_list=[loss])
            return loss_

        loss1 = run(aot_choose_kernel=True)
        loss2 = run(aot_choose_kernel=False)

        self.assertEqual(loss1, loss2)


if __name__ == "__main__":
    unittest.main()
