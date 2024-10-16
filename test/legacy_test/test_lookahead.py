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

import unittest

import numpy as np

import paddle
from paddle import base, nn
from paddle.base.framework import in_pir_mode

LOOKAHEAD_K = 5
LOOKAHEAD_ALPHA = 0.2
SGD_LR = 1.0


class TestLookAhead(unittest.TestCase):

    def test_lookahead_static(self):
        paddle.enable_static()
        place = base.CPUPlace()
        shape = [2, 3, 8, 8]
        exe = base.Executor(place)
        train_program = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup):
            with base.unique_name.guard():
                data = paddle.static.data(
                    name='X', shape=[None, 1], dtype='float32'
                )
                hidden = paddle.nn.Linear(1, 10)
                loss = paddle.mean(hidden(data))

                optimizer = paddle.optimizer.SGD(learning_rate=SGD_LR)
                lookahead = paddle.incubate.optimizer.LookAhead(
                    optimizer, alpha=LOOKAHEAD_ALPHA, k=LOOKAHEAD_K
                )
                lookahead.minimize(loss)

        exe.run(startup)
        slow_param = None
        fast_param = None
        for i in range(10):
            if (i + 1) % LOOKAHEAD_K == 0:
                slow_param = slow_param + LOOKAHEAD_ALPHA * (
                    fast_param - slow_param
                )
            x = np.random.random(size=(10, 1)).astype('float32')
            if in_pir_mode():
                for op in train_program.global_block().ops:
                    if op.name() == 'pd_op.add_grad':
                        bias_grad = op.result(1)
                fetch_list = [hidden.bias, bias_grad]
            else:
                fetch_list = [
                    'linear_0.b_0',
                    'linear_0.b_0@GRAD',
                ]
            latest_b, b_grad = exe.run(
                program=train_program,
                feed={'X': x},
                fetch_list=fetch_list,
            )
            if i == 0:
                slow_param = latest_b
            if (i + 1) % LOOKAHEAD_K == 0:
                self.assertAlmostEqual(
                    slow_param.all(), latest_b.all(), delta=5e-3
                )
            fast_param = latest_b - SGD_LR * b_grad

    def test_look_ahead_dygraph(self):
        BATCH_SIZE = 16
        BATCH_NUM = 4
        EPOCH_NUM = 4

        IMAGE_SIZE = 784
        CLASS_NUM = 10

        # define a random dataset
        class RandomDataset(paddle.io.Dataset):
            def __init__(self, num_samples):
                self.num_samples = num_samples

            def __getitem__(self, idx):
                image = np.random.random([IMAGE_SIZE]).astype('float32')
                label = np.random.randint(0, CLASS_NUM - 1, (1,)).astype(
                    'int64'
                )
                return image, label

            def __len__(self):
                return self.num_samples

        class LinearNet(nn.Layer):
            def __init__(self):
                super().__init__()
                self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
                self.bias = self._linear.bias

            @paddle.jit.to_static
            def forward(self, x):
                return self._linear(x)

        def train(layer, loader, loss_fn, opt):
            idx = 0
            slow_param = None
            fast_param = None
            for epoch_id in range(EPOCH_NUM):
                for batch_id, (image, label) in enumerate(loader()):
                    idx += 1
                    out = layer(image)
                    loss = loss_fn(out, label)
                    loss.backward()
                    fast_param = (
                        layer.bias.numpy() - SGD_LR * layer.bias.grad.numpy()
                    )
                    opt.step()
                    if idx == 1:
                        slow_param = fast_param
                    if idx % LOOKAHEAD_K == 0:
                        slow_param = slow_param + LOOKAHEAD_ALPHA * (
                            fast_param - slow_param
                        )
                        self.assertAlmostEqual(
                            np.mean(slow_param),
                            np.mean(layer.bias.numpy()),
                            delta=5e-3,
                        )
                    opt.clear_grad()

        layer = LinearNet()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.SGD(
            learning_rate=SGD_LR, parameters=layer.parameters()
        )
        lookahead = paddle.incubate.optimizer.LookAhead(
            optimizer, alpha=LOOKAHEAD_ALPHA, k=LOOKAHEAD_K
        )

        # create data loader
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        loader = paddle.io.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        )

        train(layer, loader, loss_fn, lookahead)


if __name__ == "__main__":
    unittest.main()
