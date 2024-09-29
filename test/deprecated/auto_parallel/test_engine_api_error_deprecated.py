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

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn, static
from paddle.distributed.fleet import auto
from paddle.io import Dataset

paddle.enable_static()


epoch_num = 1
batch_size = 2
batch_num = 10
hidden_size = 1024
sequence_len = 512
image_size = hidden_size
class_num = 10

is_fetch = True
is_feed = True
my_feed_vars = []


class TrainDataset(Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=image_size).astype("float32")
        label = np.random.randint(0, class_num - 1, dtype="int64")
        return input, label

    def __len__(self):
        return self.num_samples


class TestDataset(Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=image_size).astype("float32")
        return input

    def __len__(self):
        return self.num_samples


class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        dropout_ratio=0.1,
        initializer_range=0.02,
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )
        bias_attr = None

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr
        )
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr
        )
        self.linear2 = nn.Linear(d_model, 1, weight_attr, bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout = nn.Dropout(dropout_ratio, mode="upscale_in_train")

    def forward(self, input):
        out = self.norm(input)
        out = self.linear0(out)

        if is_feed:
            my_feed_vars.append((out, out.shape))

        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.linear2(out)

        if is_feed:
            my_feed_vars.append((out, out.shape))
        if is_fetch:
            auto.fetch(out, "my_fetch", logging=True)
        return out


class TestEngineErrorRaise(unittest.TestCase):
    def setUp(self):
        class NoSupportData1:
            def __getitem__(self, index):
                input = np.random.uniform(size=image_size).astype("float32")
                label = np.random.randint(0, class_num - 1, dtype="int64")
                return input, label

        class NoSupportData2(TrainDataset):
            def __getitem__(self, index):
                input = [
                    list(np.random.uniform(size=image_size).astype("float32"))
                ]
                label = [np.random.randint(0, class_num - 1, dtype="int64")]
                return input, label

        class NoSupportData3:
            def __getitem__(self, index):
                input = np.random.uniform(size=image_size).astype("float32")
                return input

        class NoSupportData4(TestDataset):
            def __getitem__(self, index):
                input = [
                    list(np.random.uniform(size=image_size).astype("float32"))
                ]
                return input

        self.no_support_data_1 = NoSupportData1()
        self.no_support_data_2 = NoSupportData2(10)
        self.no_support_data_3 = NoSupportData3()
        self.no_support_data_4 = NoSupportData4(10)

    def test_Engine(self):
        with self.assertRaises(TypeError):
            auto.Engine(model=paddle.static.Program())
        with self.assertRaises(TypeError):
            auto.Engine(loss="CrossEntropyLoss")
        with self.assertRaises(TypeError):
            auto.Engine(optimizer="adam")
        with self.assertRaises(TypeError):
            auto.Engine(metrics=["acc"])
        with self.assertRaises(TypeError):
            auto.Engine(cluster="cluster")
        with self.assertRaises(TypeError):
            auto.Engine(strategy="strategy")

    def test_fit(self):
        with self.assertRaises(TypeError):
            engine = auto.Engine(
                model=MLPLayer(),
                loss=paddle.nn.CrossEntropyLoss(),
                optimizer=paddle.optimizer.AdamW(0.00001),
            )
            engine.fit(train_data=self.no_support_data_1)

        with self.assertRaises(TypeError):
            engine = auto.Engine(
                model=MLPLayer(),
                loss=paddle.nn.CrossEntropyLoss(),
                optimizer=paddle.optimizer.AdamW(0.00001),
            )
            engine.fit(train_data=self.no_support_data_2)

    def test_evaluate(self):
        with self.assertRaises(TypeError):
            engine = auto.Engine(
                model=MLPLayer(),
                loss=paddle.nn.CrossEntropyLoss(),
                metrics=paddle.metric.Accuracy(),
            )
            engine.evaluate(valid_data=self.no_support_data_3)

        with self.assertRaises(TypeError):
            engine = auto.Engine(
                model=MLPLayer(),
                loss=paddle.nn.CrossEntropyLoss(),
                metrics=paddle.metric.Accuracy(),
            )
            engine.evaluate(
                valid_data=self.no_support_data_4, valid_sample_split=1
            )

    def test_predict(self):
        with self.assertRaises(TypeError):
            engine = auto.Engine(model=MLPLayer())
            engine.predict(
                test_data=self.no_support_data_3, test_sample_split=1
            )

        with self.assertRaises(TypeError):
            engine = auto.Engine(model=MLPLayer())
            engine.predict(
                test_data=self.no_support_data_4, test_sample_split=1
            )

    def build_program(self):
        main_prog = static.Program()
        startup_prog = static.Program()
        with static.program_guard(main_prog, startup_prog):
            input = static.data(
                name="input",
                shape=[batch_size // 2, image_size],
                dtype='float32',
            )
            label = static.data(
                name="label", shape=[batch_size // 2, 1], dtype='int64'
            )
            mlp = MLPLayer()
            loss = paddle.nn.CrossEntropyLoss()
            predict = mlp(input)
            loss_var = loss(predict, label)
        return main_prog, startup_prog, input, label, loss_var

    def test_prepare(self):
        with self.assertRaises(ValueError):
            engine = auto.Engine(model=MLPLayer())
            engine.prepare()

        with self.assertRaises(AssertionError):
            engine = auto.Engine(model=MLPLayer())
            engine.prepare(mode="train")

        with self.assertRaises(TypeError):
            input = static.data(
                name="input",
                shape=[batch_size / 2, image_size],
                dtype='float32',
            )
            label = static.data(
                name="label", shape=[batch_size / 2, 1], dtype='int64'
            )
            engine = auto.Engine(model=MLPLayer())
            engine.prepare(inputs_spec=input, labels_spec=label, mode="eval")

        input_spec = static.InputSpec(
            shape=[batch_size, image_size], dtype="float32", name="input"
        )
        label_spec = static.InputSpec(
            shape=[batch_size, image_size], dtype="float32", name="input"
        )
        (
            main_prog,
            startup_prog,
            input_var,
            label_var,
            loss_var,
        ) = self.build_program()

        with self.assertRaises(TypeError):
            engine = auto.Engine(loss=loss_var)
            engine.prepare(
                inputs=input_spec,
                labels=label_spec,
                main_program=main_prog,
                startup_program=startup_prog,
                mode="eval",
            )

        with self.assertRaises(AssertionError):
            engine = auto.Engine(loss=loss_var)
            engine.prepare(
                inputs_spec=[input_spec, input_spec],
                labels_spec=[label_spec, label_spec],
                inputs=input_var,
                labels=label_var,
                main_program=main_prog,
                startup_program=startup_prog,
                mode="predict",
            )

    def test_cost(self):
        with self.assertRaises(ValueError):
            engine = auto.Engine(model=MLPLayer())
            engine.cost(mode="predict")


class TestEngineDynamicErrorRaise(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_cost(self):
        with self.assertRaises(ValueError):
            engine = auto.Engine(model=MLPLayer())
            engine.cost(mode="predict")


if __name__ == "__main__":
    unittest.main()
