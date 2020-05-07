# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import time
import random
import tempfile
import shutil

from paddle.incubate.hapi.model import Input
from paddle.incubate.hapi.vision.models import LeNet
from paddle.incubate.hapi.callbacks import config_callbacks


class TestCallbacks(unittest.TestCase):
    def setUp(self):
        self.save_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.save_dir)

    def run_callback(self):
        epochs = 2
        steps = 50
        freq = 2
        eval_steps = 20

        lenet = LeNet()
        inputs = [Input([None, 1, 28, 28], 'float32', name='image')]
        lenet.prepare(inputs=inputs)

        cbks = config_callbacks(
            model=lenet,
            batch_size=128,
            epochs=epochs,
            steps=steps,
            log_freq=freq,
            verbose=self.verbose,
            metrics=['loss', 'acc'],
            save_dir=self.save_dir)
        cbks.on_begin('train')

        logs = {'loss': 50.341673, 'acc': 0.00256}
        for epoch in range(epochs):
            cbks.on_epoch_begin(epoch)
            for step in range(steps):
                cbks.on_batch_begin('train', step, logs)
                logs['loss'] -= random.random() * 0.1
                logs['acc'] += random.random() * 0.1
                time.sleep(0.005)
                cbks.on_batch_end('train', step, logs)
            cbks.on_epoch_end(epoch, logs)

            eval_logs = {'eval_loss': 20.341673, 'eval_acc': 0.256}
            params = {
                'steps': eval_steps,
                'metrics': ['eval_loss', 'eval_acc'],
            }
            cbks.on_begin('eval', params)
            for step in range(eval_steps):
                cbks.on_batch_begin('eval', step, eval_logs)
                eval_logs['eval_loss'] -= random.random() * 0.1
                eval_logs['eval_acc'] += random.random() * 0.1
                eval_logs['batch_size'] = 2
                time.sleep(0.005)
                cbks.on_batch_end('eval', step, eval_logs)
            cbks.on_end('eval', eval_logs)

            test_logs = {}
            params = {'steps': eval_steps}
            cbks.on_begin('test', params)
            for step in range(eval_steps):
                cbks.on_batch_begin('test', step, test_logs)
                test_logs['batch_size'] = 2
                time.sleep(0.005)
                cbks.on_batch_end('test', step, test_logs)
            cbks.on_end('test', test_logs)

        cbks.on_end('train')

    def test_callback_verbose_0(self):
        self.verbose = 0
        self.run_callback()

    def test_callback_verbose_1(self):
        self.verbose = 1
        self.run_callback()

    def test_callback_verbose_2(self):
        self.verbose = 2
        self.run_callback()


if __name__ == '__main__':
    unittest.main()
