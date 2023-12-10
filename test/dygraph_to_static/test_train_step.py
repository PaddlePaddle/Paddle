# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest
from functools import partial

import numpy as np
from dygraph_to_static_utils import Dy2StTestBase, test_ast_only, test_pt_only

import paddle


def reset_seed():
    paddle.seed(1010)
    np.random.seed(1010)
    random.seed(1010)


def loss_fn_tiny_model(x):
    return x.mean()


def train_step_tiny_model(net, x, loss_fn, opt):
    out = net(x)
    loss = loss_fn(out)
    loss.backward()
    opt.step()
    opt.clear_grad()
    return loss


class TinyModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.layer1 = paddle.nn.Linear(10, 10)

    def forward(self, data):
        return self.layer1(data)


class TestTrainStepTinyModel(Dy2StTestBase):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = lambda: 0.001
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 5
        self.rtol = 1e-4

    def get_train_step_losses(self, func, steps):
        losses = []
        net = self.net_creator()
        lr = self.lr_creator()
        optimizer = self.optimizer_creator(
            learning_rate=lr, parameters=net.parameters()
        )
        for _ in range(steps):
            loss = func(net, self.input, self.loss_fn, optimizer)
            if isinstance(lr, paddle.optimizer.lr.ReduceOnPlateau):
                lr.step(loss)
            elif isinstance(lr, paddle.optimizer.lr.LRScheduler):
                lr.step()
            losses.append(loss)
        return losses

    @test_ast_only
    @test_pt_only
    def test_train_step(self):
        reset_seed()
        dygraph_losses = self.get_train_step_losses(
            self.train_step_func, self.steps
        )
        reset_seed()
        static_func = paddle.jit.to_static(
            self.train_step_func, full_graph=True
        )
        static_losses = self.get_train_step_losses(static_func, self.steps)
        self.assertEqual(len(dygraph_losses), len(static_losses))
        for dygraph_loss, static_loss in zip(dygraph_losses, static_losses):
            dygraph_loss = dygraph_loss.numpy()
            static_loss = static_loss.numpy()
            np.testing.assert_allclose(
                dygraph_loss, static_loss, rtol=self.rtol
            )


class TestTrainStepTinyModelAdadelta(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = lambda: 0.001
        self.optimizer_creator = paddle.optimizer.Adadelta
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelAdagrad(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = lambda: 0.001
        self.optimizer_creator = paddle.optimizer.Adagrad
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelAdam(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = lambda: 0.001
        self.optimizer_creator = paddle.optimizer.Adam
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelAdamax(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = lambda: 0.001
        self.optimizer_creator = paddle.optimizer.Adamax
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelAdamW(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = lambda: 0.001
        self.optimizer_creator = paddle.optimizer.AdamW
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLamb(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = lambda: 0.001
        self.optimizer_creator = partial(
            paddle.optimizer.Lamb, lamb_weight_decay=0.01
        )
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelMomentum(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = lambda: 0.001
        self.optimizer_creator = paddle.optimizer.Momentum
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelRMSProp(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = lambda: 0.001
        self.optimizer_creator = paddle.optimizer.RMSProp
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRNoamDecay(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.NoamDecay, d_model=0.01, warmup_steps=100
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRPiecewiseDecay(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.PiecewiseDecay,
            boundaries=[3, 6, 9],
            values=[0.1, 0.2, 0.3, 0.4],
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRNaturalExpDecay(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.NaturalExpDecay,
            learning_rate=0.5,
            gamma=0.1,
        )
        self.optimizer_creator = partial(paddle.optimizer.SGD)
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRInverseTimeDecay(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.InverseTimeDecay, learning_rate=0.5, gamma=0.1
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRPolynomialDecay(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.PolynomialDecay,
            learning_rate=0.5,
            decay_steps=20,
        )
        self.optimizer_creator = paddle.optimizer.SGD

        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRLinearWarmup(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.LinearWarmup,
            learning_rate=0.5,
            warmup_steps=2,
            start_lr=0,
            end_lr=0.5,
        )
        self.optimizer_creator = partial(paddle.optimizer.SGD)
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRExponentialDecay(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.ExponentialDecay, learning_rate=0.5, gamma=0.9
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRMultiStepDecay(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.MultiStepDecay,
            learning_rate=0.5,
            milestones=[2, 4, 6],
            gamma=0.8,
        )
        self.optimizer_creator = paddle.optimizer.SGD

        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRStepDecay(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.StepDecay,
            learning_rate=0.5,
            step_size=5,
            gamma=0.8,
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRLambdaDecay(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.LambdaDecay,
            learning_rate=0.5,
            lr_lambda=lambda x: 0.95**x,
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRReduceOnPlateau(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.ReduceOnPlateau,
            learning_rate=1.0,
            factor=0.5,
            patience=5,
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRCosineAnnealingDecay(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.CosineAnnealingDecay,
            learning_rate=0.5,
            T_max=10,
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRMultiplicativeDecay(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.MultiplicativeDecay,
            learning_rate=0.5,
            lr_lambda=lambda x: 0.95,
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLROneCycleLR(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.OneCycleLR, max_learning_rate=1.0, total_steps=3
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelLRCyclicLR(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.CyclicLR,
            base_learning_rate=0.5,
            max_learning_rate=1.0,
            step_size_up=15,
            step_size_down=5,
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


class TestTrainStepTinyModelCosineAnnealingWarmRestarts(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([10000, 10])
        self.net_creator = TinyModel
        self.lr_creator = partial(
            paddle.optimizer.lr.CosineAnnealingWarmRestarts,
            learning_rate=0.5,
            T_0=1,
            T_mult=1,
        )
        self.optimizer_creator = paddle.optimizer.SGD
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4


if __name__ == "__main__":
    unittest.main()
