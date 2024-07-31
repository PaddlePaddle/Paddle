#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial

import numpy as np
from op_test_ipu import IPUOpTest

import paddle
import paddle.optimizer
import paddle.static


class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_attrs()
        self.set_optimizer()

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['model_path'] = tempfile.TemporaryDirectory()

    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.SGD, learning_rate=1e-1)

    @IPUOpTest.static_graph
    def build_model(self):
        generator = paddle.base.unique_name.UniqueNameGenerator()
        with paddle.base.unique_name.guard(generator):
            x = paddle.static.data(
                name=self.feed_list[0],
                shape=self.feed_shape[0],
                dtype='float32',
            )
            conv1 = paddle.nn.Conv2D(
                in_channels=x.shape[1],
                out_channels=3,
                kernel_size=3,
                bias_attr=False,
            )(x)

            loss = paddle.mean(conv1)
            # apply optimizer
            self.optimizer().minimize(loss)
            self.fetch_list = [loss]

    def run_model(self, exec_mode, save_otherwise_load):
        self.build_model()

        place = paddle.IPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(self.startup_prog)

        if not save_otherwise_load:
            paddle.static.load(self.main_prog, self.attrs['model_path'].name)

        ipu_strategy = paddle.static.IpuStrategy()
        ipu_strategy.set_graph_config(is_training=True)
        if self.is_fp16_mode(exec_mode):
            ipu_strategy.set_precision_config(enable_fp16=True)
            IPUOpTest.cast_model_to_fp16(self.main_prog)
        ipu_compiler = paddle.static.IpuCompiledProgram(
            self.main_prog, ipu_strategy=ipu_strategy
        )
        program = ipu_compiler.compile(self.feed_list, self.fetch_list)

        feed = self.feed_fp32
        if self.is_fp16_mode(exec_mode):
            feed = self.feed_fp16

        result = []
        run_steps = (
            self.attrs['steps']
            if save_otherwise_load
            else self.attrs['steps'] - self.attrs['save_at_step']
        )
        for i in range(run_steps):
            tmp = exe.run(program, feed=feed, fetch_list=self.fetch_list)

            if save_otherwise_load and i == self.attrs['save_at_step'] - 1:
                ipu_compiler._backend.weights_to_host()
                paddle.static.save(
                    self.main_prog, self.attrs['model_path'].name
                )

            if save_otherwise_load and i >= self.attrs['save_at_step']:
                result.append(tmp)
            elif not save_otherwise_load:
                result.append(tmp)

        return np.asarray(result)

    def test_base(self):
        res0 = self.run_model(IPUOpTest.ExecutionMode.IPU_FP32, True)
        res1 = self.run_model(IPUOpTest.ExecutionMode.IPU_FP32, False)
        np.testing.assert_allclose(
            res0.flatten(), res1.flatten(), rtol=1e-05, atol=self.atol
        )
        self.attrs['model_path'].cleanup()


class TestMomentum(TestBase):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Momentum, learning_rate=1e-1)


class TestAdam(TestBase):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Adam, learning_rate=1e-1)


class TestLamb(TestBase):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Lamb, learning_rate=1e-1)


class TestAdamW(TestBase):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.AdamW, learning_rate=1e-1)


class TestAdamax(TestBase):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Adamax, learning_rate=1e-1)


class TestAdagrad(TestBase):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Adagrad, learning_rate=1e-1)


class TestAdadelta(TestBase):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Adagrad, learning_rate=1e-1)


class TestRMSProp(TestBase):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.RMSProp, learning_rate=1e-1)


class TestCenteredRMSProp(TestBase):
    def set_optimizer(self):
        self.optimizer = partial(
            paddle.optimizer.RMSProp, learning_rate=1e-1, centered=True
        )


@unittest.skipIf(IPUOpTest.use_ipumodel(), "skip for ipumodel")
class TestSGDFP16(TestBase):
    def set_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['model_path'] = tempfile.TemporaryDirectory()

    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.SGD, learning_rate=1e-1)

    def test_base(self):
        res0 = self.run_model(IPUOpTest.ExecutionMode.IPU_FP16, True)
        res1 = self.run_model(IPUOpTest.ExecutionMode.IPU_FP16, False)
        np.testing.assert_allclose(
            res0.flatten(), res1.flatten(), rtol=1e-05, atol=self.atol
        )
        self.attrs['model_path'].cleanup()


class TestMomentumFp16(TestSGDFP16):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Momentum, learning_rate=1e-1)


class TestAdamFP16(TestSGDFP16):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Adam, learning_rate=1e-1)


class TestLambFP16(TestSGDFP16):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Lamb, learning_rate=1e-1)


class TestAdamWFP16FP16(TestSGDFP16):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.AdamW, learning_rate=1e-1)


class TestAdamaxFP16(TestSGDFP16):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Adamax, learning_rate=1e-1)


class TestAdagradFP16(TestSGDFP16):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Adagrad, learning_rate=1e-1)


class TestAdadeltaFP16(TestSGDFP16):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.Adagrad, learning_rate=1e-1)


class TestRMSPropFP16(TestSGDFP16):
    def set_optimizer(self):
        self.optimizer = partial(paddle.optimizer.RMSProp, learning_rate=1e-1)


class TestCenteredRMSPropFP16(TestSGDFP16):
    def set_optimizer(self):
        self.optimizer = partial(
            paddle.optimizer.RMSProp, learning_rate=1e-1, centered=True
        )


if __name__ == "__main__":
    unittest.main()
