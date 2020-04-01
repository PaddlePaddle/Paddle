# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import unittest


class TestBCELoss(unittest.TestCase):
    def test_BCELoss_mean(self):
        input_np = np.random.random(size=(20, 30)).astype(np.float32)
        label_np = np.random.random(size=(20, 30)).astype(np.float32)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[None, 30], dtype='float32')
            label = fluid.data(name='label', shape=[None, 30], dtype='float32')
            bce_loss = paddle.nn.loss.BCELoss()
            res = bce_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(
                prog,
                feed={"input": input_np,
                      "label": label_np},
                fetch_list=[res])

        with fluid.dygraph.guard():
            bce_loss = paddle.nn.loss.BCELoss()
            dy_res = bce_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        expected = np.mean(-1 * (label_np * np.log(input_np) +
                                 (1. - label_np) * np.log(1. - input_np)))
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_BCELoss_sum(self):
        input_np = np.random.random(size=(20, 30)).astype(np.float32)
        label_np = np.random.random(size=(20, 30)).astype(np.float32)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[None, 30], dtype='float32')
            label = fluid.data(name='label', shape=[None, 30], dtype='float32')
            bce_loss = paddle.nn.loss.BCELoss(reduction='sum')
            res = bce_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(
                prog,
                feed={"input": input_np,
                      "label": label_np},
                fetch_list=[res])

        with fluid.dygraph.guard():
            bce_loss = paddle.nn.loss.BCELoss(reduction='sum')
            dy_res = bce_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        expected = np.sum(-1 * (label_np * np.log(input_np) +
                                (1. - label_np) * np.log(1. - input_np)))
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_BCELoss_weight(self):
        input_np = np.random.random(size=(20, 30)).astype(np.float32)
        label_np = np.random.random(size=(20, 30)).astype(np.float32)
        weight_np = np.random.random(size=(20, 30)).astype(np.float32)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[None, 30], dtype='float32')
            label = fluid.data(name='label', shape=[None, 30], dtype='float32')
            weight = fluid.data(
                name='weight', shape=[None, 30], dtype='float32')
            bce_loss = paddle.nn.loss.BCELoss(weight=weight)
            res = bce_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            bce_loss = paddle.nn.loss.BCELoss(
                weight=fluid.dygraph.to_variable(weight_np))
            dy_res = bce_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        expected = np.mean(-1 * weight_np *
                           (label_np * np.log(input_np) +
                            (1. - label_np) * np.log(1. - input_np)))
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))


if __name__ == "__main__":
    unittest.main()
if __name__ == "__main__":
    unittest.main()
