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

from __future__ import division
from __future__ import print_function

import unittest

import os
import numpy as np
import shutil
import tempfile

from paddle import fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.dygraph.container import Sequential
from paddle.fluid.dygraph.base import to_variable

from paddle.incubate.hapi.model import Model, Input, set_device
from paddle.incubate.hapi.loss import CrossEntropy
from paddle.incubate.hapi.metrics import Accuracy
from paddle.incubate.hapi.datasets import MNIST
from paddle.incubate.hapi.vision.models import LeNet
from paddle.incubate.hapi.distributed import DistributedBatchSampler, prepare_distributed_context


class LeNetDygraph(fluid.dygraph.Layer):
    def __init__(self, num_classes=10, classifier_activation='softmax'):
        super(LeNetDygraph, self).__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2D(
                1, 6, 3, stride=1, padding=1),
            Pool2D(2, 'max', 2),
            Conv2D(
                6, 16, 5, stride=1, padding=0),
            Pool2D(2, 'max', 2))

        if num_classes > 0:
            self.fc = Sequential(
                Linear(400, 120),
                Linear(120, 84),
                Linear(
                    84, 10, act=classifier_activation))

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = fluid.layers.flatten(x, 1)
            x = self.fc(x)
        return x


class MnistDataset(MNIST):
    def __init__(self, mode, return_label=True, sample_num=None):
        super(MnistDataset, self).__init__(mode=mode)
        self.return_label = return_label
        if sample_num:
            self.images = self.images[:sample_num]
            self.labels = self.labels[:sample_num]

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = np.reshape(img, [1, 28, 28])
        if self.return_label:
            return img, np.array(self.labels[idx]).astype('int64')
        return img,

    def __len__(self):
        return len(self.images)


def compute_acc(pred, label):
    pred = np.argmax(pred, -1)
    label = np.array(label)
    correct = pred[:, np.newaxis] == label
    return np.sum(correct) / correct.shape[0]


def dynamic_train(model, dataloader):
    optim = fluid.optimizer.Adam(
        learning_rate=0.001, parameter_list=model.parameters())
    model.train()
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = fluid.layers.cross_entropy(outputs, labels)
        avg_loss = fluid.layers.reduce_sum(loss)
        avg_loss.backward()
        optim.minimize(avg_loss)
        model.clear_gradients()


def dynamic_evaluate(model, dataloader):
    with fluid.dygraph.no_grad():
        model.eval()
        cnt = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)

            cnt += (np.argmax(outputs.numpy(), -1)[:, np.newaxis] ==
                    labels.numpy()).astype('int').sum()

    return cnt / len(dataloader.dataset)


@unittest.skipIf(not fluid.is_compiled_with_cuda(),
                 'CPU testing is not supported')
class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not fluid.is_compiled_with_cuda():
            self.skipTest('module not tested when ONLY_CPU compling')
        cls.device = set_device('gpu')
        fluid.enable_dygraph(cls.device)

        sp_num = 1280
        cls.train_dataset = MnistDataset(mode='train', sample_num=sp_num)
        cls.val_dataset = MnistDataset(mode='test', sample_num=sp_num)
        cls.test_dataset = MnistDataset(
            mode='test', return_label=False, sample_num=sp_num)

        cls.train_loader = fluid.io.DataLoader(
            cls.train_dataset, places=cls.device, batch_size=64)
        cls.val_loader = fluid.io.DataLoader(
            cls.val_dataset, places=cls.device, batch_size=64)
        cls.test_loader = fluid.io.DataLoader(
            cls.test_dataset, places=cls.device, batch_size=64)

        seed = 333
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        dy_lenet = LeNetDygraph()
        cls.init_param = dy_lenet.state_dict()
        dynamic_train(dy_lenet, cls.train_loader)

        cls.acc1 = dynamic_evaluate(dy_lenet, cls.val_loader)

        cls.inputs = [Input([-1, 1, 28, 28], 'float32', name='image')]
        cls.labels = [Input([None, 1], 'int64', name='label')]

        cls.save_dir = tempfile.mkdtemp()
        cls.weight_path = os.path.join(cls.save_dir, 'lenet')
        fluid.dygraph.save_dygraph(dy_lenet.state_dict(), cls.weight_path)

        fluid.disable_dygraph()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.save_dir)

    def test_fit_dygraph(self):
        self.fit(True)

    def test_fit_static(self):
        self.fit(False)

    def test_evaluate_dygraph(self):
        self.evaluate(True)

    def test_evaluate_static(self):
        self.evaluate(False)

    def test_predict_dygraph(self):
        self.predict(True)

    def test_predict_static(self):
        self.predict(False)

    def test_prepare_context(self):
        prepare_distributed_context()

    def fit(self, dynamic):
        fluid.enable_dygraph(self.device) if dynamic else None
        seed = 333
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        model = LeNet()
        optim_new = fluid.optimizer.Adam(
            learning_rate=0.001, parameter_list=model.parameters())
        model.prepare(
            optim_new,
            loss_function=CrossEntropy(average=False),
            metrics=Accuracy(),
            inputs=self.inputs,
            labels=self.labels)
        model.fit(self.train_dataset, batch_size=64, shuffle=False)

        result = model.evaluate(self.val_dataset, batch_size=64)
        np.testing.assert_allclose(result['acc'], self.acc1)

        train_sampler = DistributedBatchSampler(
            self.train_dataset, batch_size=64, shuffle=False)
        val_sampler = DistributedBatchSampler(
            self.val_dataset, batch_size=64, shuffle=False)

        train_loader = fluid.io.DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            places=self.device,
            return_list=True)

        val_loader = fluid.io.DataLoader(
            self.val_dataset,
            batch_sampler=val_sampler,
            places=self.device,
            return_list=True)

        model.fit(train_loader, val_loader)
        fluid.disable_dygraph() if dynamic else None

    def evaluate(self, dynamic):
        fluid.enable_dygraph(self.device) if dynamic else None
        model = LeNet()
        model.prepare(
            metrics=Accuracy(), inputs=self.inputs, labels=self.labels)
        model.load(self.weight_path)
        result = model.evaluate(self.val_dataset, batch_size=64)
        np.testing.assert_allclose(result['acc'], self.acc1)

        sampler = DistributedBatchSampler(
            self.val_dataset, batch_size=64, shuffle=False)

        val_loader = fluid.io.DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            places=self.device,
            return_list=True)

        model.evaluate(val_loader)

        fluid.disable_dygraph() if dynamic else None

    def predict(self, dynamic):
        fluid.enable_dygraph(self.device) if dynamic else None
        model = LeNet()
        model.prepare(inputs=self.inputs)
        model.load(self.weight_path)
        output = model.predict(
            self.test_dataset, batch_size=64, stack_outputs=True)
        np.testing.assert_equal(output[0].shape[0], len(self.test_dataset))

        acc = compute_acc(output[0], self.val_dataset.labels)
        np.testing.assert_allclose(acc, self.acc1)

        sampler = DistributedBatchSampler(
            self.test_dataset, batch_size=64, shuffle=False)

        test_loader = fluid.io.DataLoader(
            self.test_dataset,
            batch_sampler=sampler,
            places=self.device,
            return_list=True)

        model.evaluate(test_loader)

        fluid.disable_dygraph() if dynamic else None


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self._fc = Linear(20, 10, act='softmax')

    def forward(self, x):
        y = self._fc(x)
        return y


class TestModelFunction(unittest.TestCase):
    def set_seed(self, seed=1024):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

    def test_train_batch(self, dynamic=True):
        dim = 20
        data = np.random.random(size=(4, dim)).astype(np.float32)
        label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)

        def get_expect():
            fluid.enable_dygraph(fluid.CPUPlace())
            self.set_seed()
            m = MyModel()
            optim = fluid.optimizer.SGD(learning_rate=0.001,
                                        parameter_list=m.parameters())
            m.train()
            output = m(to_variable(data))
            l = to_variable(label)
            loss = fluid.layers.cross_entropy(output, l)
            avg_loss = fluid.layers.reduce_sum(loss)
            avg_loss.backward()
            optim.minimize(avg_loss)
            m.clear_gradients()
            fluid.disable_dygraph()
            return avg_loss.numpy()

        ref = get_expect()
        for dynamic in [True, False]:
            device = set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            self.set_seed()
            model = MyModel()

            optim2 = fluid.optimizer.SGD(learning_rate=0.001,
                                         parameter_list=model.parameters())

            inputs = [Input([None, dim], 'float32', name='x')]
            labels = [Input([None, 1], 'int64', name='label')]
            model.prepare(
                optim2,
                loss_function=CrossEntropy(average=False),
                inputs=inputs,
                labels=labels,
                device=device)
            loss, = model.train_batch([data], [label])

            np.testing.assert_allclose(loss.flatten(), ref.flatten())
            fluid.disable_dygraph() if dynamic else None

    def test_test_batch(self, dynamic=True):
        dim = 20
        data = np.random.random(size=(4, dim)).astype(np.float32)

        def get_expect():
            fluid.enable_dygraph(fluid.CPUPlace())
            self.set_seed()
            m = MyModel()
            m.eval()
            output = m(to_variable(data))
            fluid.disable_dygraph()
            return output.numpy()

        ref = get_expect()
        for dynamic in [True, False]:
            device = set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            self.set_seed()
            model = MyModel()
            inputs = [Input([None, dim], 'float32', name='x')]
            model.prepare(inputs=inputs, device=device)
            out, = model.test_batch([data])

            np.testing.assert_allclose(out, ref)
            fluid.disable_dygraph() if dynamic else None

    def test_save_load(self):
        path = tempfile.mkdtemp()
        for dynamic in [True, False]:
            device = set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            model = MyModel()
            inputs = [Input([None, 20], 'float32', name='x')]
            labels = [Input([None, 1], 'int64', name='label')]
            optim = fluid.optimizer.SGD(learning_rate=0.001,
                                        parameter_list=model.parameters())
            model.prepare(
                inputs=inputs,
                optimizer=optim,
                loss_function=CrossEntropy(average=False),
                labels=labels)
            model.save(path + '/test')
            model.load(path + '/test')
            shutil.rmtree(path)
            fluid.disable_dygraph() if dynamic else None

    def test_dynamic_save_static_load(self):
        path = tempfile.mkdtemp()
        # for dynamic in [True, False]:
        device = set_device('cpu')
        fluid.enable_dygraph(device)  #if dynamic else None
        model = MyModel()
        inputs = [Input([None, 20], 'float32', name='x')]
        labels = [Input([None, 1], 'int64', name='label')]
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=model.parameters())
        model.prepare(
            inputs=inputs,
            optimizer=optim,
            loss_function=CrossEntropy(average=False),
            labels=labels)
        model.save(path + '/test')
        fluid.disable_dygraph()
        model = MyModel()
        inputs = [Input([None, 20], 'float32', name='x')]
        labels = [Input([None, 1], 'int64', name='label')]
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=model.parameters())
        model.prepare(
            inputs=inputs,
            optimizer=optim,
            loss_function=CrossEntropy(average=False),
            labels=labels)
        model.load(path + '/test')
        shutil.rmtree(path)

    def test_static_save_dynamic_load(self):
        path = tempfile.mkdtemp()

        model = MyModel()
        inputs = [Input([None, 20], 'float32', name='x')]
        labels = [Input([None, 1], 'int64', name='label')]
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=model.parameters())
        model.prepare(
            inputs=inputs,
            optimizer=optim,
            loss_function=CrossEntropy(average=False),
            labels=labels)
        model.save(path + '/test')

        device = set_device('cpu')
        fluid.enable_dygraph(device)  #if dynamic else None

        model = MyModel()
        inputs = [Input([None, 20], 'float32', name='x')]
        labels = [Input([None, 1], 'int64', name='label')]
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=model.parameters())
        model.prepare(
            inputs=inputs,
            optimizer=optim,
            loss_function=CrossEntropy(average=False),
            labels=labels)
        model.load(path + '/test')
        shutil.rmtree(path)
        fluid.disable_dygraph()

    def test_parameters(self):
        for dynamic in [True, False]:
            device = set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            model = MyModel()
            inputs = [Input([None, 20], 'float32', name='x')]
            model.prepare(inputs=inputs)
            params = model.parameters()
            self.assertTrue(params[0].shape[0] == 20)
            self.assertTrue(params[0].shape[1] == 10)
            fluid.disable_dygraph() if dynamic else None

    def test_export_deploy_model(self):
        model = LeNet()
        inputs = [Input([-1, 1, 28, 28], 'float32', name='image')]
        model.prepare(inputs=inputs)
        save_dir = tempfile.mkdtemp()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        tensor_img = np.array(
            np.random.random((1, 1, 28, 28)), dtype=np.float32)
        ori_results = model.test_batch(tensor_img)

        model.save_inference_model(save_dir)

        place = fluid.CPUPlace() if not fluid.is_compiled_with_cuda(
        ) else fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        [inference_program, feed_target_names, fetch_targets] = (
            fluid.io.load_inference_model(
                dirname=save_dir, executor=exe))

        results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)

        np.testing.assert_allclose(results, ori_results)
        shutil.rmtree(save_dir)


if __name__ == '__main__':
    unittest.main()
