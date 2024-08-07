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

import unittest

import numpy as np
from op_test_ipu import IPUOpTest

import paddle
import paddle.static


class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_attrs()

    def set_atol(self):
        self.atol = 1e-6

    def set_data_feed(self):
        self.feed = {
            "image": np.random.uniform(size=[1, 3, 10, 10]).astype('float32'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [x.dtype for x in self.feed.values()]

    def set_attrs(self):
        self.attrs = {
            "optimizer": 'sgd',
            "weight_decay": 0.0,
            "loss_scaling": 1.0,
        }

    def _test_optimizer(self, run_ipu=True):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        paddle.seed(self.SEED)
        np.random.seed(self.SEED)

        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                image = paddle.static.data(
                    name='image', shape=[1, 3, 10, 10], dtype='float32'
                )
                conv1 = paddle.nn.Conv2D(
                    in_channels=image.shape[1],
                    out_channels=3,
                    kernel_size=3,
                    bias_attr=False,
                )(image)

                loss = paddle.mean(conv1)

                weight_decay = self.attrs['weight_decay']
                opt = paddle.optimizer.SGD(
                    learning_rate=1e-1, weight_decay=weight_decay
                )
                if self.attrs['optimizer'] == 'adam':
                    opt = paddle.optimizer.Adam(
                        learning_rate=1e-1, weight_decay=weight_decay
                    )
                elif self.attrs['optimizer'] == 'lamb':
                    opt = paddle.optimizer.Lamb(
                        learning_rate=1e-1, lamb_weight_decay=weight_decay
                    )
                opt.minimize(loss)

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = [image.name]
                fetch_list = [loss.name]
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=True)
                ipu_strategy.set_options(
                    {'loss_scaling': self.attrs["loss_scaling"]}
                )
                if "use_no_bias_optimizer" in self.attrs.keys():
                    ipu_strategy.set_options(
                        {
                            "use_no_bias_optimizer": self.attrs[
                                "use_no_bias_optimizer"
                            ]
                        }
                    )
                if "accl1_type" in self.attrs.keys():
                    ipu_strategy.set_options(
                        {"accl1_type": self.attrs["accl1_type"]}
                    )
                program = paddle.static.IpuCompiledProgram(
                    main_prog, ipu_strategy=ipu_strategy
                ).compile(feed_list, fetch_list)
            else:
                program = main_prog

            result = []
            for epoch in range(100):
                loss_res = exe.run(program, feed=self.feed, fetch_list=[loss])
                result.append(loss_res)

            return np.array(result)

    def test(self):
        # cpu and ipu dimension mismatch, cpu:(100, 1, 1), ipu:(100, 1)
        ipu_loss = self._test_optimizer(True).flatten()
        cpu_loss = self._test_optimizer(False).flatten()

        np.testing.assert_allclose(
            ipu_loss, cpu_loss, rtol=1e-05, atol=self.atol
        )


@unittest.skip('do not support L2 regularization')
class TestSGD(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'sgd',
            "weight_decay": 0.1,
            "loss_scaling": 2.0,
        }


@unittest.skip('do not support L2 regularization')
class TestAdamCase1(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'adam',
            "weight_decay": 0.1,
            "loss_scaling": 3.0,
        }


class TestAdamCase2(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'adam',
            "weight_decay": 0.0,
            "loss_scaling": 4.0,
        }


@unittest.skip('cpu do not support AdamNoBias')
class TestAdamNoBias(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'adam',
            "weight_decay": 0.0,
            "loss_scaling": 4.0,
            "use_no_bias_optimizer": True,
        }


@unittest.skip('cpu do not support FLOAT16')
class TestAdamCase3(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'adam',
            "weight_decay": 0.0,
            "loss_scaling": 4.0,
            "accl1_type": "FLOAT16",
        }


@unittest.skip('seems cpu output wrong')
class TestLambCase1(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'lamb',
            "weight_decay": 0.0,
            "loss_scaling": 5.0,
        }


@unittest.skip('seems cpu output wrong')
class TestLamb(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'lamb',
            "weight_decay": 0.1,
            "loss_scaling": 6.0,
        }


@unittest.skip('cpu do not support LambNoBias')
class TestLambNoBias(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'lamb',
            "weight_decay": 0.1,
            "loss_scaling": 6.0,
            "use_no_bias_optimizer": True,
        }


@unittest.skip('cpu do not support FLOAT16')
class TestLambCase2(TestBase):
    def set_attrs(self):
        self.attrs = {
            "optimizer": 'lamb',
            "weight_decay": 0.1,
            "loss_scaling": 6.0,
            "accl1_type": "FLOAT16",
        }


if __name__ == "__main__":
    unittest.main()
