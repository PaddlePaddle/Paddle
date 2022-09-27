# copyright (c) 2020 paddlepaddle authors. all rights reserved.
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

os.environ['FLAGS_cudnn_deterministic'] = '1'

import unittest
import tempfile
import numpy as np

import paddle
from paddle import fluid

from paddle import Model
from paddle.static import InputSpec
from paddle.nn.layer.loss import CrossEntropyLoss
from paddle.vision.models import LeNet
from paddle.vision.datasets import MNIST
import paddle.vision.transforms as T


@unittest.skipIf(not fluid.is_compiled_with_cuda(),
                 'CPU testing is not supported')
class TestHapiWithAmp(unittest.TestCase):

    def get_model(self, amp_config):
        net = LeNet()
        inputs = InputSpec([None, 1, 28, 28], "float32", 'x')
        labels = InputSpec([None, 1], "int64", "y")
        model = Model(net, inputs, labels)
        optim = paddle.optimizer.Adam(learning_rate=0.001,
                                      parameters=model.parameters())
        model.prepare(optimizer=optim,
                      loss=CrossEntropyLoss(reduction="sum"),
                      amp_configs=amp_config)
        return model

    def run_model(self, model):
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = MNIST(mode='train', transform=transform)
        model.fit(train_dataset,
                  epochs=1,
                  batch_size=64,
                  num_iters=2,
                  log_freq=1)

    def run_amp(self, amp_level):
        for dynamic in [True, False]:
            if not dynamic and amp_level['level'] == 'O2':
                amp_level['use_fp16_guard'] = False
            print('dynamic' if dynamic else 'static', amp_level)

            paddle.seed(2021)
            paddle.enable_static() if not dynamic else paddle.disable_static()
            paddle.set_device('gpu')
            model = self.get_model(amp_level)
            self.run_model(model)

    def test_pure_fp16(self):
        amp_config = {
            "level": "O2",
            "init_loss_scaling": 128,
        }
        self.run_amp(amp_config)

    def test_amp(self):
        amp_config = {"level": "O1", "init_loss_scaling": 128}
        self.run_amp(amp_config)

    def test_fp32(self):
        amp_config = {
            "level": "O0",
        }
        self.run_amp(amp_config)

    def test_save_load(self):
        paddle.disable_static()
        paddle.set_device('gpu')
        amp_level = {"level": "O1", "init_loss_scaling": 128}
        paddle.seed(2021)
        model = self.get_model(amp_level)
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = MNIST(mode='train', transform=transform)
        model.fit(train_dataset,
                  epochs=1,
                  batch_size=64,
                  num_iters=2,
                  log_freq=1)
        temp_dir = tempfile.TemporaryDirectory()
        lenet_amp_path = os.path.join(temp_dir.name, './lenet_amp')
        model.save(lenet_amp_path)

        with paddle.fluid.unique_name.guard():
            paddle.seed(2021)
            new_model = self.get_model(amp_level)
            train_dataset = MNIST(mode='train', transform=transform)
            new_model.fit(train_dataset,
                          epochs=1,
                          batch_size=64,
                          num_iters=1,
                          log_freq=1)
        # not equal before load
        self.assertNotEqual(new_model._scaler.state_dict()['incr_count'],
                            model._scaler.state_dict()['incr_count'])
        print((new_model._scaler.state_dict()['incr_count'],
               model._scaler.state_dict()['incr_count']))

        # equal after load
        new_model.load(lenet_amp_path)
        temp_dir.cleanup()
        self.assertEqual(new_model._scaler.state_dict()['incr_count'],
                         model._scaler.state_dict()['incr_count'])
        self.assertEqual(new_model._scaler.state_dict()['decr_count'],
                         model._scaler.state_dict()['decr_count'])
        np.testing.assert_array_equal(
            new_model._optimizer.state_dict()['conv2d_1.w_0_moment1_0'].numpy(),
            model._optimizer.state_dict()['conv2d_1.w_0_moment1_0'].numpy())

    def test_dynamic_check_input(self):
        paddle.disable_static()
        amp_configs_list = [
            {
                "level": "O3"
            },
            {
                "level": "O1",
                "test": 0
            },
            {
                "level": "O1",
                "use_fp16_guard": True
            },
            "O3",
        ]
        if not fluid.is_compiled_with_cuda():
            self.skipTest('module not tested when ONLY_CPU compling')
        paddle.set_device('gpu')
        net = LeNet()
        model = Model(net)
        optim = paddle.optimizer.Adam(learning_rate=0.001,
                                      parameters=model.parameters())
        loss = CrossEntropyLoss(reduction="sum")
        with self.assertRaises(ValueError):
            for amp_configs in amp_configs_list:
                model.prepare(optimizer=optim,
                              loss=loss,
                              amp_configs=amp_configs)
        model.prepare(optimizer=optim, loss=loss, amp_configs="O2")
        model.prepare(optimizer=optim,
                      loss=loss,
                      amp_configs={
                          "custom_white_list": {"matmul"},
                          "init_loss_scaling": 1.0
                      })

    def test_static_check_input(self):
        paddle.enable_static()
        amp_configs = {"level": "O2", "use_pure_fp16": True}
        if not fluid.is_compiled_with_cuda():
            self.skipTest('module not tested when ONLY_CPU compling')
        paddle.set_device('gpu')

        net = LeNet()
        inputs = InputSpec([None, 1, 28, 28], "float32", 'x')
        labels = InputSpec([None, 1], "int64", "y")
        model = Model(net, inputs, labels)

        optim = paddle.optimizer.Adam(learning_rate=0.001,
                                      parameters=model.parameters())
        loss = CrossEntropyLoss(reduction="sum")
        with self.assertRaises(ValueError):
            model.prepare(optimizer=optim, loss=loss, amp_configs=amp_configs)


if __name__ == '__main__':
    unittest.main()
