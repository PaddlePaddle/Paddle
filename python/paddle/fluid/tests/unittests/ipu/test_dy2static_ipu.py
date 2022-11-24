#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import unittest

import numpy as np
import paddle
<<<<<<< HEAD
from paddle.fluid.dygraph.dygraph_to_static.program_translator import ProgramCache
=======
from paddle.fluid.dygraph.dygraph_to_static.program_translator import (
    ProgramCache,
)
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUD2STest
from paddle.jit import to_static
from paddle.optimizer.lr import LRScheduler
from functools import partial


class SimpleLayer(paddle.nn.Layer):
<<<<<<< HEAD

    def __init__(self,
                 loss_op=None,
                 use_softmax=True,
                 use_reduction=True,
                 use_identity_loss=True):
        super(SimpleLayer, self).__init__()
        self.loss_op = loss_op
        self.conv = paddle.nn.Conv2D(in_channels=3,
                                     out_channels=1,
                                     kernel_size=2,
                                     stride=1)
=======
    def __init__(
        self,
        loss_op=None,
        use_softmax=True,
        use_reduction=True,
        use_identity_loss=True,
    ):
        super().__init__()
        self.loss_op = loss_op
        self.conv = paddle.nn.Conv2D(
            in_channels=3, out_channels=1, kernel_size=2, stride=1
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
        self.use_softmax = use_softmax
        self.use_reduction = use_reduction
        self.use_identity_loss = use_identity_loss

    @to_static()
    def forward(self, x, target=None):
        x = self.conv(x)
        x = paddle.fluid.layers.flatten(x, axis=1)
        if target is not None:
            if self.use_softmax:
                x = paddle.fluid.layers.softmax(x)
            if self.loss_op:
                loss = self.loss_op(x, target)
            else:
                loss = paddle.fluid.layers.cross_entropy(x, target)
            if self.use_reduction:
                loss = paddle.mean(loss)
            if self.use_identity_loss:
                loss = paddle.incubate.identity_loss(loss, 1)
            return x, loss
        return x


class TestBase(IPUD2STest):
<<<<<<< HEAD

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def setUp(self):
        self.set_op_attrs()
        self.set_data_feed()

    def set_op_attrs(self):
        self.loss_op = paddle.fluid.layers.cross_entropy

    def set_data_feed(self):
        self.data = paddle.uniform((8, 3, 10, 10), dtype='float32')
        self.label = paddle.randint(0, 10, shape=[8], dtype='int64')

    def create_model(self, use_ipu=False):
<<<<<<< HEAD
        return SimpleLayer(loss_op=self.loss_op,
                           use_softmax=True,
                           use_reduction=not use_ipu,
                           use_identity_loss=use_ipu)
=======
        return SimpleLayer(
            loss_op=self.loss_op,
            use_softmax=True,
            use_reduction=not use_ipu,
            use_identity_loss=use_ipu,
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    def _test(self, use_ipu=False):
        paddle.seed(self.SEED)
        np.random.seed(self.SEED)
        model = self.create_model(use_ipu)
<<<<<<< HEAD
        optim = paddle.optimizer.Adam(learning_rate=0.01,
                                      parameters=model.parameters())
=======
        optim = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=model.parameters()
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

        if use_ipu:
            paddle.set_device('ipu')
            ipu_strategy = paddle.static.IpuStrategy()
<<<<<<< HEAD
            ipu_strategy.set_graph_config(num_ipus=1,
                                          is_training=True,
                                          micro_batch_size=1,
                                          enable_manual_shard=False)
=======
            ipu_strategy.set_graph_config(
                num_ipus=1,
                is_training=True,
                micro_batch_size=1,
                enable_manual_shard=False,
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            ipu_strategy.set_optimizer(optim)

        epochs = 100
        result = []
        for _ in range(epochs):
            # ipu only needs call model() to do forward/backward/grad_update
            pred, loss = model(self.data, self.label)
            if not use_ipu:
                loss.backward()
                optim.step()
                optim.clear_grad()
            result.append(loss)

        if use_ipu:
            ipu_strategy.release_patch()

        return np.array(result)

    def test_training(self):
        ipu_loss = self._test(True).flatten()
        cpu_loss = self._test(False).flatten()
<<<<<<< HEAD
        self.assertTrue(np.allclose(ipu_loss, cpu_loss, atol=1e-4))


class TestSaveLoad(TestBase):

=======
        np.testing.assert_allclose(ipu_loss, cpu_loss, rtol=1e-05, atol=1e-4)


class TestSaveLoad(TestBase):
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def setUp(self):
        super().setUp()
        self.save_path = tempfile.TemporaryDirectory()

    def tearDown(self):
        super().tearDown()
        self.save_path.cleanup()

    def _test(self, use_ipu=False):
        paddle.seed(self.SEED)
        np.random.seed(self.SEED)
        model = self.create_model(use_ipu)
<<<<<<< HEAD
        optim = paddle.optimizer.Adam(learning_rate=0.01,
                                      parameters=model.parameters())
        model_path = '{}/model_state_dict_{}.pdparams'.format(
            self.save_path, 'ipu' if use_ipu else 'cpu')
        optim_path = '{}/optim_state_dict_{}.pdopt'.format(
            self.save_path, 'ipu' if use_ipu else 'cpu')
=======
        optim = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=model.parameters()
        )
        model_path = '{}/model_state_dict_{}.pdparams'.format(
            self.save_path, 'ipu' if use_ipu else 'cpu'
        )
        optim_path = '{}/optim_state_dict_{}.pdopt'.format(
            self.save_path, 'ipu' if use_ipu else 'cpu'
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

        if use_ipu:
            paddle.set_device('ipu')
            ipu_strategy = paddle.static.IpuStrategy()
<<<<<<< HEAD
            ipu_strategy.set_graph_config(num_ipus=1,
                                          is_training=True,
                                          micro_batch_size=1,
                                          enable_manual_shard=False)
=======
            ipu_strategy.set_graph_config(
                num_ipus=1,
                is_training=True,
                micro_batch_size=1,
                enable_manual_shard=False,
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            ipu_strategy.set_optimizer(optim)

        epochs = 100
        result = []
        for _ in range(epochs):
            # ipu only needs call model() to do forward/backward/grad_update
            pred, loss = model(self.data, self.label)
            if not use_ipu:
                loss.backward()
                optim.step()
                optim.clear_grad()
            result.append(loss)

        if use_ipu:
            paddle.fluid.core.IpuBackend.get_instance().weights_to_host()

        paddle.save(model.state_dict(), model_path)
        paddle.save(optim.state_dict(), optim_path)
        model.set_state_dict(paddle.load(model_path))
        optim.set_state_dict(paddle.load(optim_path))

        for _ in range(epochs):
            # ipu only needs call model() to do forward/backward/grad_update
            pred, loss = model(self.data, self.label)
            if not use_ipu:
                loss.backward()
                optim.step()
                optim.clear_grad()
            result.append(loss)

        if use_ipu:
            ipu_strategy.release_patch()

        return np.array(result)


class TestPatch(IPUD2STest):
<<<<<<< HEAD

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def setUp(cls):
        paddle.disable_static()

    def test(self, use_ipu=False):
        old_getter = ProgramCache.__getitem__
        old_step = LRScheduler.step

        ipu_strategy = paddle.static.IpuStrategy()
        ipu_strategy.release_patch()

        reset_getter = ProgramCache.__getitem__
        reset_step = LRScheduler.step

        self.assertTrue(reset_getter is old_getter)
        self.assertTrue(reset_step is old_step)


class TestWithoutIdentityLoss1(TestBase):
<<<<<<< HEAD

    def create_model(self, use_ipu=False):
        return SimpleLayer(loss_op=self.loss_op,
                           use_softmax=True,
                           use_reduction=True,
                           use_identity_loss=False)


class TestWithoutIdentityLoss2(TestBase):

=======
    def create_model(self, use_ipu=False):
        return SimpleLayer(
            loss_op=self.loss_op,
            use_softmax=True,
            use_reduction=True,
            use_identity_loss=False,
        )


class TestWithoutIdentityLoss2(TestBase):
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def set_op_attrs(self):
        self.loss_op = paddle.fluid.layers.softmax_with_cross_entropy

    def set_data_feed(self):
        self.data = paddle.uniform((8, 3, 10, 10), dtype='float32')
        self.label = paddle.randint(0, 10, shape=[8, 1], dtype='int64')

    def create_model(self, use_ipu=False):
<<<<<<< HEAD
        return SimpleLayer(loss_op=self.loss_op,
                           use_softmax=False,
                           use_reduction=True,
                           use_identity_loss=False)


class TestWithoutIdentityLoss3(TestBase):

=======
        return SimpleLayer(
            loss_op=self.loss_op,
            use_softmax=False,
            use_reduction=True,
            use_identity_loss=False,
        )


class TestWithoutIdentityLoss3(TestBase):
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def set_op_attrs(self):
        self.loss_op = partial(paddle.fluid.layers.kldiv_loss, reduction="none")

    def set_data_feed(self):
        self.data = paddle.uniform((8, 3, 10, 10), dtype='float32')
        self.label = paddle.rand(shape=[8, 81], dtype='float32')

    def create_model(self, use_ipu=False):
<<<<<<< HEAD
        return SimpleLayer(loss_op=self.loss_op,
                           use_softmax=True,
                           use_reduction=True,
                           use_identity_loss=False)


class TestWithoutIdentityLoss4(TestBase):

=======
        return SimpleLayer(
            loss_op=self.loss_op,
            use_softmax=True,
            use_reduction=True,
            use_identity_loss=False,
        )


class TestWithoutIdentityLoss4(TestBase):
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def set_op_attrs(self):
        self.loss_op = paddle.nn.functional.binary_cross_entropy

    def set_data_feed(self):
        self.data = paddle.uniform((8, 3, 10, 10), dtype='float32')
        self.label = paddle.rand(shape=[8, 81], dtype='float32')

    def create_model(self, use_ipu=False):
<<<<<<< HEAD
        return SimpleLayer(loss_op=self.loss_op,
                           use_softmax=True,
                           use_reduction=False,
                           use_identity_loss=False)


class TestWithoutIdentityLoss5(TestBase):

=======
        return SimpleLayer(
            loss_op=self.loss_op,
            use_softmax=True,
            use_reduction=False,
            use_identity_loss=False,
        )


class TestWithoutIdentityLoss5(TestBase):
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def set_op_attrs(self):
        self.loss_op = paddle.fluid.layers.sigmoid_cross_entropy_with_logits

    def set_data_feed(self):
        self.data = paddle.uniform((8, 3, 10, 10), dtype='float32')
<<<<<<< HEAD
        self.label = paddle.randint(0, 10, shape=[8, 81],
                                    dtype='int64').astype('float32')

    def create_model(self, use_ipu=False):
        return SimpleLayer(loss_op=self.loss_op,
                           use_softmax=True,
                           use_reduction=True,
                           use_identity_loss=False)
=======
        self.label = paddle.randint(0, 10, shape=[8, 81], dtype='int64').astype(
            'float32'
        )

    def create_model(self, use_ipu=False):
        return SimpleLayer(
            loss_op=self.loss_op,
            use_softmax=True,
            use_reduction=True,
            use_identity_loss=False,
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f


if __name__ == "__main__":
    unittest.main()
