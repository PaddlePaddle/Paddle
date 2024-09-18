# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

sys.path.append("../../auto_parallel")

from test_to_static_deprecated import MLPLayer, MyDataset

import paddle
from paddle.distributed.fleet import auto

paddle.enable_static()


class TestEngineBase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.batch_num = 5
        self.hidden_size = 1024

        self.init_model()
        self.init_optimizer()
        self.init_dataset()
        self.init_engine()

    def init_model(self):
        self.mlp = MLPLayer(
            hidden_size=self.hidden_size,
            intermediate_size=4 * self.hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02,
        )
        self.loss = paddle.nn.CrossEntropyLoss()

    def init_optimizer(self):
        self.optimizer = paddle.optimizer.SGD(
            learning_rate=0.00001, parameters=self.mlp.parameters()
        )

    def init_dataset(self):
        self.dataset = MyDataset(self.batch_num * self.batch_size)

    def init_engine(self):
        # inputs = InputSpec([self.batch_size, self.hidden_size], 'float32', 'x')
        # labels = InputSpec([self.batch_size], 'int64', 'label')

        self.engine = auto.Engine(
            model=self.mlp,
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=paddle.metric.Accuracy(),
        )


class TestLRScheduler(TestEngineBase):
    def init_optimizer(self):
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=0.00001, T_max=10
        )
        self.optimizer = paddle.optimizer.SGD(learning_rate=scheduler)

    def test_lr_scheduler(self):
        self.init_engine()
        self.engine.fit(self.dataset, batch_size=self.batch_size)
        lr = self.engine._optimizer._learning_rate
        assert isinstance(lr, paddle.optimizer.lr.LRScheduler)


class TestGradClipByGlobalNorm(TestEngineBase):
    def init_optimizer(self):
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        self.optimizer = paddle.optimizer.SGD(
            learning_rate=0.00001, grad_clip=clip
        )

    def test_grad_clip(self):
        self.engine.fit(self.dataset, batch_size=self.batch_size)
        self.check_program()

    def check_program(self):
        ops = self.engine.main_program.global_block().ops
        has_grad_clip = False
        for op in ops:
            if op.desc.has_attr("op_namescope") and op.desc.attr(
                "op_namescope"
            ).startswith("/gradient_clip"):
                has_grad_clip = True
                break
        assert has_grad_clip is True


class TestGradClipByNorm(TestGradClipByGlobalNorm):
    def init_optimizer(self):
        clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
        self.optimizer = paddle.optimizer.SGD(
            learning_rate=0.00001, grad_clip=clip
        )


if __name__ == "__main__":
    unittest.main()
