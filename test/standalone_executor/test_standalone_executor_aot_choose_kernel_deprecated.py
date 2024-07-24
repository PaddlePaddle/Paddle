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


def build_resnet50(use_amp=False):
    with paddle.pir_utils.OldIrGuard():
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dtype = 'float16' if use_amp else 'float32'
        with paddle.static.program_guard(main_program, startup_program):
            image = paddle.static.data(
                name='image', shape=[32, 3, 224, 224], dtype=dtype
            )
            label = paddle.static.data(name='label', shape=[32], dtype='int64')
            model = paddle.vision.models.resnet50()
            prediction = model(image)
            loss = paddle.nn.functional.cross_entropy(
                input=prediction, label=label
            )
            loss = paddle.mean(loss)
            adam = paddle.optimizer.Adam(learning_rate=0.001)

            if use_amp:
                adam = paddle.static.amp.decorate(
                    optimizer=adam,
                    init_loss_scaling=1.0,
                    use_dynamic_loss_scaling=False,
                    use_pure_fp16=True,
                    use_fp16_guard=False,
                )
            adam.minimize(loss)

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.enable_addto = True
        build_strategy.fuse_elewise_add_act_ops = True
        if use_amp:
            build_strategy.fuse_bn_act_ops = True
            build_strategy.fuse_bn_add_act_ops = True

        main_program = paddle.static.CompiledProgram(
            main_program, build_strategy=build_strategy
        )

    return main_program, startup_program, loss, adam


def run_resnet50(aot_choose_kernel=False, use_amp=False):
    with paddle.pir_utils.OldIrGuard():
        paddle.seed(2022)
        np.random.seed(2022)

        main_program, startup_program, loss, optimizer = build_resnet50(use_amp)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        scope = paddle.static.Scope()

        set_flags({'FLAGS_cudnn_deterministic': 1})
        if aot_choose_kernel:
            set_flags({'FLAGS_new_executor_static_build': 1})

        if use_amp:
            set_flags({'FLAGS_conv_workspace_size_limit': 1500})
            set_flags({'FLAGS_max_inplace_grad_add': 8})
            set_flags({'FLAGS_cudnn_batchnorm_spatial_persistent': 1})

        with paddle.static.scope_guard(scope):
            exe.run(startup_program)
            if use_amp:
                optimizer.amp_init(place)

            feed_dtype = 'float16' if use_amp else 'float32'
            for i in range(1):
                feed = {
                    'image': np.random.randint(
                        0, 256, size=[32, 3, 224, 224]
                    ).astype(feed_dtype),
                    'label': np.random.randint(0, 1000, size=[32]).astype(
                        'int64'
                    ),
                }
                loss_ = exe.run(main_program, feed=feed, fetch_list=[loss])
    return loss_


class TestAOTChooseKernel(unittest.TestCase):
    def test_resnet50_aot_choose_kernel(self):
        if not paddle.base.core.is_compiled_with_cuda():
            return
        loss1 = run_resnet50(aot_choose_kernel=True)
        loss2 = run_resnet50(aot_choose_kernel=False)
        self.assertEqual(loss1, loss2)

    def test_resnet50_amp_aot_choose_kernel(self):
        if not paddle.base.core.is_compiled_with_cuda():
            return
        loss1 = run_resnet50(aot_choose_kernel=True, use_amp=True)
        loss2 = run_resnet50(aot_choose_kernel=False, use_amp=True)
        self.assertEqual(loss1, loss2)


if __name__ == "__main__":
    unittest.main()
