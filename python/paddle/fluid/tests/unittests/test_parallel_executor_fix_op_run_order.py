# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import unittest
import numpy as np
from paddle.vision.models import resnet50
from paddle.nn import CrossEntropyLoss


class TestFixOpRunOrder(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        if paddle.is_compiled_with_cuda():
            fluid.set_flags({'FLAGS_cudnn_deterministic': 1})

    def get_place(self):
        return paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()

    def get_feed(self):
        batch_size = 32
        image = np.random.random([batch_size, 3, 224, 224]).astype('float32')
        label = np.random.randint(0, 1000, [batch_size, 1]).astype('int64')
        return {"image": image, "label": label}

    def create_model(self, fix_op_run_order):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        scope = paddle.static.Scope()
        with paddle.static.program_guard(main_prog, startup_prog):
            image = paddle.static.data(
                name="image", shape=[None, 3, 224, 224], dtype="float32")
            label = paddle.static.data(
                name="label", shape=[None, 1], dtype="int64")
            model = resnet50()
            pred = model(image)
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(pred, label)
            optimizer = paddle.optimizer.SGD(learning_rate=1e-3)
            optimizer.minimize(loss)

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.fix_op_run_order = fix_op_run_order
        build_strategy.fuse_bn_act_ops = True
        build_strategy.fuse_bn_add_act_ops = True
        main_prog = paddle.static.CompiledProgram(main_prog).with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            places=[self.get_place()])

        exe = paddle.static.Executor(self.get_place())
        with paddle.static.scope_guard(scope):
            exe.run(startup_prog)

        return main_prog, scope, loss

    def run_and_fetch_loss(self, main_prog, scope, loss, feed):
        with paddle.static.scope_guard(scope):
            exe = paddle.static.Executor(self.get_place())
            loss_value = exe.run(main_prog, feed=feed, fetch_list=[loss])[0]
            return loss_value

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return

        main1, scope1, loss1 = self.create_model(True)
        main2, scope2, loss2 = self.create_model(False)
        for i in range(10):
            feed = self.get_feed()
            loss_val1 = self.run_and_fetch_loss(main1, scope1, loss1, feed)
            loss_val2 = self.run_and_fetch_loss(main2, scope2, loss2, feed)
            self.assertEqual(loss_val1, loss_val2)


if __name__ == "__main__":
    unittest.main()
