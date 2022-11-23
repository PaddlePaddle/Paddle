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

import logging
import numpy as np
import os
import paddle
import shutil
import tempfile
import unittest

paddle.enable_static()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger("paddle_with_cinn")


def set_cinn_flag(val):
    cinn_compiled = False
    try:
        paddle.set_flags({'FLAGS_use_cinn': val})
        cinn_compiled = True
    except ValueError:
        logger.warning("The used paddle is not compiled with CINN.")
    return cinn_compiled


def reader(limit):
    for _ in range(limit):
        yield np.random.random([1, 28]).astype('float32'), \
            np.random.randint(0, 2, size=[1]).astype('int64')


def rand_data(img, label, loop_num=10):
    feed = []
    data = reader(loop_num)
    for _ in range(loop_num):
        d, l = next(data)
        feed.append({img: d, label: l})
    return feed


def build_program(main_program, startup_program):
    with paddle.static.program_guard(main_program, startup_program):
        img = paddle.static.data(name='img', shape=[1, 28], dtype='float32')
        param = paddle.create_parameter(
            name="bias",
            shape=[1, 28],
            dtype="float32",
            attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(
                np.random.rand(1, 28).astype(np.float32))))
        label = paddle.static.data(name="label", shape=[1], dtype='int64')

        hidden = paddle.add(img, param)
        prediction = paddle.nn.functional.relu(hidden)

        loss = paddle.nn.functional.cross_entropy(input=prediction, label=label)
        avg_loss = paddle.mean(loss)
        adam = paddle.optimizer.Adam(learning_rate=0.001)
        adam.minimize(avg_loss)
    return img, label, avg_loss


def train(dot_save_dir, prefix, seed=1234):
    np.random.seed(seed)
    paddle.seed(seed)
    if paddle.is_compiled_with_cuda():
        paddle.set_flags({'FLAGS_cudnn_deterministic': 1})

    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    img, label, loss = build_program(main_program, startup_program)

    place = paddle.CUDAPlace(
        0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.debug_graphviz_path = os.path.join(dot_save_dir, prefix)
    compiled_program = paddle.static.CompiledProgram(
        main_program, build_strategy).with_data_parallel(loss_name=loss.name)

    iters = 100
    feed = rand_data(img.name, label.name, iters)
    loss_values = []
    for step in range(iters):
        loss_v = exe.run(compiled_program, feed=feed[step], fetch_list=[loss])
        loss_values.append(loss_v[0][0])
    return loss_values


@unittest.skipIf(not set_cinn_flag(True), "Paddle is not compiled with CINN.")
class TestParallelExecutorRunCinn(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dots_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_run_with_cinn(self):
        cinn_losses = train(self.tmpdir, "paddle")
        set_cinn_flag(False)
        pd_losses = train(self.tmpdir, "cinn")
        np.testing.assert_allclose(cinn_losses,
                                   pd_losses,
                                   rtol=1e-05,
                                   atol=1e-05)


if __name__ == '__main__':
    unittest.main()
