#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.framework import core
from paddle.nn.quant import quant_layers

paddle.enable_static()


def init_data(batch_size=32, img_shape=[784], label_range=9):
    np.random.seed(5)
    assert isinstance(img_shape, list)
    input_shape = [batch_size, *img_shape]
    img = np.random.random(size=input_shape).astype(np.float32)
    label = (
        np.array([np.random.randint(0, label_range) for _ in range(batch_size)])
        .reshape((-1, 1))
        .astype("int64")
    )
    return img, label


class TestMovingAverageAbsMaxScaleOp(unittest.TestCase):
    def check_backward(self, use_cuda):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            image = paddle.static.data(
                name='image', shape=[-1, 784], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )
            fc_tmp = paddle.static.nn.fc(image, size=10, activation='softmax')
            out_scale = quant_layers.MovingAverageAbsMaxScale(
                name=fc_tmp.name, dtype=fc_tmp.dtype
            )
            fc_tmp_1 = out_scale(fc_tmp)
            cross_entropy = paddle.nn.functional.cross_entropy(fc_tmp, label)
            loss = paddle.mean(cross_entropy)
            sgd = paddle.optimizer.SGD(learning_rate=1e-3)
            sgd.minimize(loss)

        moving_average_abs_max_scale_ops = [
            op
            for op in main_program.blocks[0].ops
            if op.type == 'moving_average_abs_max_scale'
        ]
        assert (
            len(moving_average_abs_max_scale_ops) == 1
        ), "The number of moving_average_abs_max_scale_ops should be 1."

        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(startup_program)

        binary = paddle.static.CompiledProgram(main_program)

        img, label = init_data()
        feed_dict = {"image": img, "label": label}
        res = exe.run(binary, feed_dict)

    def test_check_op_times(self):
        if core.is_compiled_with_cuda():
            self.check_backward(use_cuda=True)
        self.check_backward(use_cuda=False)


if __name__ == '__main__':
    unittest.main()
