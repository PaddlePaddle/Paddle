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
import unittest
import numpy as np
from time import time
from test_mnist import MNIST, SEED, TestMNIST

if paddle.fluid.is_compiled_with_cuda():
    paddle.fluid.set_flags({'FLAGS_cudnn_deterministic': True})


class TestPureFP16(TestMNIST):

    def train_static(self):
        return self.train(to_static=True)

    def train_dygraph(self):
        return self.train(to_static=False)

    def test_mnist_to_static(self):
        if paddle.fluid.is_compiled_with_cuda():
            dygraph_loss = self.train_dygraph()
            static_loss = self.train_static()
            # NOTE: In pure fp16 training, loss is not stable, so we enlarge atol here.
<<<<<<< HEAD
            self.assertTrue(np.allclose(dygraph_loss, static_loss, atol=1e-3),
                            msg='dygraph is {}\n static_res is \n{}'.format(
                                dygraph_loss, static_loss))
=======
            np.testing.assert_allclose(
                dygraph_loss,
                static_loss,
                rtol=1e-05,
                atol=0.001,
                err_msg='dygraph is {}\n static_res is \n{}'.format(
                    dygraph_loss, static_loss
                ),
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

    def train(self, to_static=False):
        np.random.seed(SEED)
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        mnist = MNIST()

        if to_static:
            print("Successfully to apply @to_static.")
            build_strategy = paddle.static.BuildStrategy()
            # Why set `build_strategy.enable_inplace = False` here?
            # Because we find that this PASS strategy of PE makes dy2st training loss unstable.
            build_strategy.enable_inplace = False
            mnist = paddle.jit.to_static(mnist, build_strategy=build_strategy)

<<<<<<< HEAD
        optimizer = paddle.optimizer.Adam(learning_rate=0.001,
                                          parameters=mnist.parameters())

        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        mnist, optimizer = paddle.amp.decorate(models=mnist,
                                               optimizers=optimizer,
                                               level='O2',
                                               save_dtype='float32')
=======
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=mnist.parameters()
        )

        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        mnist, optimizer = paddle.amp.decorate(
            models=mnist, optimizers=optimizer, level='O2', save_dtype='float32'
        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

        loss_data = []
        for epoch in range(self.epoch_num):
            start = time()
            for batch_id, data in enumerate(self.train_reader()):
<<<<<<< HEAD
                dy_x_data = np.array([x[0].reshape(1, 28, 28)
                                      for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data
                                   ]).astype('int64').reshape(-1, 1)
=======
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]
                ).astype('float32')
                y_data = (
                    np.array([x[1] for x in data])
                    .astype('int64')
                    .reshape(-1, 1)
                )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

                img = paddle.to_tensor(dy_x_data)
                label = paddle.to_tensor(y_data)
                label.stop_gradient = True

<<<<<<< HEAD
                with paddle.amp.auto_cast(enable=True,
                                          custom_white_list=None,
                                          custom_black_list=None,
                                          level='O2'):
=======
                with paddle.amp.auto_cast(
                    enable=True,
                    custom_white_list=None,
                    custom_black_list=None,
                    level='O2',
                ):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                    prediction, acc, avg_loss = mnist(img, label=label)

                scaled = scaler.scale(avg_loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)

                loss_data.append(avg_loss.numpy()[0])
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 2 == 0:
                    print(
<<<<<<< HEAD
                        "Loss at epoch {} step {}: loss: {:}, acc: {}, cost: {}"
                        .format(epoch, batch_id, avg_loss.numpy(), acc.numpy(),
                                time() - start))
=======
                        "Loss at epoch {} step {}: loss: {:}, acc: {}, cost: {}".format(
                            epoch,
                            batch_id,
                            avg_loss.numpy(),
                            acc.numpy(),
                            time() - start,
                        )
                    )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                    start = time()
                if batch_id == 10:
                    break
        return loss_data


if __name__ == '__main__':
    with paddle.fluid.framework._test_eager_guard():
        unittest.main()
