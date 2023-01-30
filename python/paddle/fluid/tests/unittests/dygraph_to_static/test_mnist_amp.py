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

<<<<<<< HEAD
import unittest
from time import time

import numpy as np
from test_mnist import MNIST, SEED, TestMNIST

import paddle
=======
import paddle
import unittest
import numpy as np
from time import time
from test_mnist import MNIST, TestMNIST, SEED
from paddle.jit import ProgramTranslator
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from paddle.fluid.optimizer import AdamOptimizer

if paddle.fluid.is_compiled_with_cuda():
    paddle.fluid.set_flags({'FLAGS_cudnn_deterministic': True})


class TestAMP(TestMNIST):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def train_static(self):
        return self.train(to_static=True)

    def train_dygraph(self):
        return self.train(to_static=False)

    def test_mnist_to_static(self):
        dygraph_loss = self.train_dygraph()
        static_loss = self.train_static()
        # NOTE(Aurelius84): In static AMP training, there is a grep_list but
        # dygraph AMP don't. It will bring the numbers of cast_op is different
        # and leads to loss has a bit diff.
        np.testing.assert_allclose(
            dygraph_loss,
            static_loss,
            rtol=1e-05,
            atol=0.001,
            err_msg='dygraph is {}\n static_res is \n{}'.format(
<<<<<<< HEAD
                dygraph_loss, static_loss
            ),
        )
=======
                dygraph_loss, static_loss))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def train(self, to_static=False):
        paddle.seed(SEED)
        mnist = MNIST()

        if to_static:
            print("Successfully to apply @to_static.")
            mnist = paddle.jit.to_static(mnist)

<<<<<<< HEAD
        adam = AdamOptimizer(
            learning_rate=0.001, parameter_list=mnist.parameters()
        )
=======
        adam = AdamOptimizer(learning_rate=0.001,
                             parameter_list=mnist.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        loss_data = []
        for epoch in range(self.epoch_num):
            start = time()
            for batch_id, data in enumerate(self.train_reader()):
<<<<<<< HEAD
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]
                ).astype('float32')
                y_data = (
                    np.array([x[1] for x in data])
                    .astype('int64')
                    .reshape(-1, 1)
                )
=======
                dy_x_data = np.array([x[0].reshape(1, 28, 28)
                                      for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data
                                   ]).astype('int64').reshape(-1, 1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                img = paddle.to_tensor(dy_x_data)
                label = paddle.to_tensor(y_data)
                label.stop_gradient = True

                with paddle.amp.auto_cast():
                    prediction, acc, avg_loss = mnist(img, label=label)

                scaled = scaler.scale(avg_loss)
                scaled.backward()
                scaler.minimize(adam, scaled)

                loss_data.append(avg_loss.numpy()[0])
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 10 == 0:
                    print(
<<<<<<< HEAD
                        "Loss at epoch {} step {}: loss: {:}, acc: {}, cost: {}".format(
                            epoch,
                            batch_id,
                            avg_loss.numpy(),
                            acc.numpy(),
                            time() - start,
                        )
                    )
=======
                        "Loss at epoch {} step {}: loss: {:}, acc: {}, cost: {}"
                        .format(epoch, batch_id, avg_loss.numpy(), acc.numpy(),
                                time() - start))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    start = time()
                if batch_id == 50:
                    break
        return loss_data


if __name__ == '__main__':
<<<<<<< HEAD
    unittest.main()
=======
    with paddle.fluid.framework._test_eager_guard():
        unittest.main()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
