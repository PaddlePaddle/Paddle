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

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle import nn
from paddle.distributed.fleet import auto
from paddle.vision.datasets import MNIST

paddle.enable_static()

batch_size = 2
hidden_size = 1024
# sequence_len = 512
image_size = hidden_size
class_num = 10


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
        auto.shard_tensor(input, auto.ProcessMesh([0]), [None, None])
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out


class TestSaveLoad(unittest.TestCase):
    def test_fp32_save_fp16_load(self):

        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02,
        )
        loss = paddle.nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.00001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            grad_clip=None,
        )
        metric = paddle.metric.Accuracy()

        inputs_spec = [
            paddle.static.InputSpec(
                shape=[batch_size, image_size], name="input", dtype="float32"
            )
        ]
        labels_spec = [
            paddle.static.InputSpec(
                shape=[batch_size, 1], name="label", dtype="int64"
            )
        ]

        # build fp32 model
        strategy = auto.Strategy()
        strategy.auto_mode = "semi"
        engine_fp32 = auto.Engine(
            mlp, loss, optimizer, metric, strategy=strategy
        )
        engine_fp32.prepare(inputs_spec, labels_spec, mode="train")
        fp32_state = {
            k: np.array(v)
            for k, v in engine_fp32.main_program.state_dict("param").items()
        }
        # save
        temp_dir = tempfile.TemporaryDirectory()
        model_filename = os.path.join(temp_dir.name, 'mlp')
        engine_fp32.save(model_filename)

        # build fp16 model
        strategy = auto.Strategy()
        strategy.auto_mode = "semi"
        amp = strategy.amp
        amp.enable = True
        amp.dtype = "float16"
        amp.level = "o2"
        engine_fp16 = auto.Engine(
            mlp, loss, optimizer, metric, strategy=strategy
        )
        engine_fp16.load(model_filename)
        engine_fp16.prepare(inputs_spec, labels_spec, mode="train")
        fp16_state = {
            k: np.array(v)
            for k, v in engine_fp16.main_program.state_dict("param").items()
        }

        # check param
        for name, fp32_param in fp32_state.items():
            fp16_param = fp16_state[name]
            if "layer_norm" in name:
                assert fp16_param.dtype == np.float32
            else:
                assert fp16_param.dtype == np.float16
            np.testing.assert_allclose(fp32_param, fp16_param, atol=1e-4)

        temp_dir.cleanup()


class TestDistSaveLoad(unittest.TestCase):
    def setUp(self):
        self.save_dir = tempfile.mkdtemp()
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        self.train_dataset = MNIST(mode='train', transform=transform)
        self.test_dataset = MNIST(mode='test', transform=transform)
        self.prepare_engine()

    def tearDown(self):
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def prepare_engine(self):
        model = paddle.vision.models.LeNet()
        loss = paddle.nn.CrossEntropyLoss()
        base_lr = 1e-3
        boundaries = [5, 8]
        values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
        lr = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=boundaries, values=values, verbose=False
        )
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=model.parameters()
        )
        auto.fetch(model.parameters()[0], "param0", logging=True)
        metrics = paddle.metric.Accuracy(topk=(1, 2))
        self.engine = auto.Engine(model, loss, optimizer, metrics)

    def test_single_save_load(self):
        history = self.engine.fit(
            train_data=self.train_dataset,
            valid_data=self.test_dataset,
            batch_size=128,
            steps_per_epoch=360,
            valid_steps=40,
            log_freq=1,
            epochs=2,
            save_dir=self.save_dir,
            save_freq=1,
            save_checkpoint_every_n_step=10,
            keep_checkpoint_max_num=4,
            load_dir=self.save_dir,
        )

    def test_dist_save_load(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        launch_model_path = os.path.join(file_dir, "engine_load_train.py")

        if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
            coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
        else:
            coverage_args = []

        tmp_dir = tempfile.TemporaryDirectory()
        cmd = (
            [sys.executable, "-u"]
            + coverage_args
            + [
                "-m",
                "paddle.distributed.launch",
                "--devices",
                "0,1",
                "--log_dir",
                tmp_dir.name,
                launch_model_path,
            ]
        )

        process = subprocess.Popen(cmd)
        process.wait()
        self.assertEqual(process.returncode, 0)

        tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
