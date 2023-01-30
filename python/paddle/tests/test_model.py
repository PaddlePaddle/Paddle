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

<<<<<<< HEAD
import os
import shutil
import tempfile
import unittest

import numpy as np

import paddle
import paddle.jit as jit
import paddle.vision.models as models
from paddle import Model, fluid, to_tensor
from paddle.hapi.model import prepare_distributed_context
from paddle.io import Dataset, DistributedBatchSampler
from paddle.metric import Accuracy
from paddle.nn import Conv2D, Linear, ReLU, Sequential
from paddle.nn.layer.loss import CrossEntropyLoss
from paddle.static import InputSpec
from paddle.vision.datasets import MNIST
from paddle.vision.models import LeNet


class LeNetDygraph(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2D(1, 6, 3, stride=1, padding=1),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
            Conv2D(6, 16, 5, stride=1, padding=0),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
        )

        if num_classes > 0:
            self.fc = Sequential(
                Linear(400, 120), Linear(120, 84), Linear(84, 10)
            )
=======
from __future__ import division
from __future__ import print_function

import unittest

import os
import numpy as np
import shutil
import tempfile

import paddle
from paddle import fluid
from paddle import to_tensor
from paddle.nn import Conv2D, Linear, ReLU, Sequential, Softmax

from paddle import Model
from paddle.static import InputSpec
from paddle.nn.layer.loss import CrossEntropyLoss
from paddle.metric import Accuracy
from paddle.vision.datasets import MNIST
from paddle.vision.models import LeNet
import paddle.vision.models as models
import paddle.fluid.dygraph.jit as jit
from paddle.io import DistributedBatchSampler, Dataset
from paddle.hapi.model import prepare_distributed_context
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid.dygraph.dygraph_to_static.program_translator import ProgramTranslator


class LeNetDygraph(paddle.nn.Layer):

    def __init__(self, num_classes=10):
        super(LeNetDygraph, self).__init__()
        self.num_classes = num_classes
        self.features = Sequential(Conv2D(1, 6, 3, stride=1, padding=1), ReLU(),
                                   paddle.fluid.dygraph.Pool2D(2, 'max', 2),
                                   Conv2D(6, 16, 5, stride=1, padding=0),
                                   ReLU(),
                                   paddle.fluid.dygraph.Pool2D(2, 'max', 2))

        if num_classes > 0:
            self.fc = Sequential(Linear(400, 120), Linear(120, 84),
                                 Linear(84, 10))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
<<<<<<< HEAD
            x = paddle.flatten(x, 1, -1)
=======
            x = fluid.layers.flatten(x, 1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            x = self.fc(x)
        return x


class ModelInner(paddle.nn.Layer):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(ModelInner, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.fc = paddle.nn.Linear(3, 4)

    def forward(self, x):
        y = self.fc(x)
        return y, 0


class ModelOutter(paddle.nn.Layer):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(ModelOutter, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.module1 = ModelInner()
        self.module2 = paddle.nn.Linear(4, 5)

    def forward(self, x):
        y, dummpy = self.module1(x)
        y = self.module2(y)
        return y, 3


class LeNetListInput(paddle.nn.Layer):
<<<<<<< HEAD
    def __init__(self, num_classes=10):
        super().__init__()
=======

    def __init__(self, num_classes=10):
        super(LeNetListInput, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.num_classes = num_classes
        self.cov = Conv2D(1, 6, 3, stride=1, padding=1)
        for param in self.cov.parameters():
            param.trainable = False
<<<<<<< HEAD
        self.features = Sequential(
            self.cov,
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
            Conv2D(6, 16, 5, stride=1, padding=0),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
        )

        if num_classes > 0:
            self.fc = Sequential(
                Linear(400, 120), Linear(120, 84), Linear(84, 10)
            )
=======
        self.features = Sequential(self.cov, ReLU(),
                                   paddle.fluid.dygraph.Pool2D(2, 'max', 2),
                                   Conv2D(6, 16, 5, stride=1, padding=0),
                                   ReLU(),
                                   paddle.fluid.dygraph.Pool2D(2, 'max', 2))

        if num_classes > 0:
            self.fc = Sequential(Linear(400, 120), Linear(120, 84),
                                 Linear(84, 10))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def forward(self, inputs):
        x = inputs[0]
        x = self.features(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x + inputs[1])
        return x


class LeNetDictInput(LeNetDygraph):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def forward(self, inputs):
        x = self.features(inputs['x1'])

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x + inputs['x2'])
        return x


class MnistDataset(MNIST):
<<<<<<< HEAD
    def __init__(self, mode, return_label=True, sample_num=None):
        super().__init__(mode=mode)
=======

    def __init__(self, mode, return_label=True, sample_num=None):
        super(MnistDataset, self).__init__(mode=mode)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.return_label = return_label
        if sample_num:
            self.images = self.images[:sample_num]
            self.labels = self.labels[:sample_num]

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = np.reshape(img, [1, 28, 28])
        if self.return_label:
            return img, np.array(self.labels[idx]).astype('int64')
<<<<<<< HEAD
        return (img,)
=======
        return img,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def __len__(self):
        return len(self.images)


def compute_acc(pred, label):
    pred = np.argmax(pred, -1)
    label = np.array(label)
    correct = pred[:, np.newaxis] == label
    return np.sum(correct) / correct.shape[0]


def dynamic_train(model, dataloader):
<<<<<<< HEAD
    optim = fluid.optimizer.Adam(
        learning_rate=0.001, parameter_list=model.parameters()
    )
=======
    optim = fluid.optimizer.Adam(learning_rate=0.001,
                                 parameter_list=model.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    model.train()
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = CrossEntropyLoss(reduction="sum")(outputs, labels)
<<<<<<< HEAD
        avg_loss = paddle.sum(loss)
=======
        avg_loss = fluid.layers.reduce_sum(loss)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        avg_loss.backward()
        optim.minimize(avg_loss)
        model.clear_gradients()


def dynamic_evaluate(model, dataloader):
    with fluid.dygraph.no_grad():
        model.eval()
        cnt = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)

<<<<<<< HEAD
            cnt += (
                (
                    np.argmax(outputs.numpy(), -1)[:, np.newaxis]
                    == labels.numpy()
                )
                .astype('int')
                .sum()
            )
=======
            cnt += (np.argmax(
                outputs.numpy(),
                -1)[:, np.newaxis] == labels.numpy()).astype('int').sum()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    return cnt / len(dataloader.dataset)


<<<<<<< HEAD
@unittest.skipIf(
    not fluid.is_compiled_with_cuda(), 'CPU testing is not supported'
)
class TestModel(unittest.TestCase):
=======
@unittest.skipIf(not fluid.is_compiled_with_cuda(),
                 'CPU testing is not supported')
class TestModel(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @classmethod
    def setUpClass(cls):
        if not fluid.is_compiled_with_cuda():
            cls().skipTest('module not tested when ONLY_CPU compling')
        cls.device = paddle.set_device('gpu')
        fluid.enable_dygraph(cls.device)

        sp_num = 1280
        cls.train_dataset = MnistDataset(mode='train', sample_num=sp_num)
        cls.val_dataset = MnistDataset(mode='test', sample_num=sp_num)
<<<<<<< HEAD
        cls.test_dataset = MnistDataset(
            mode='test', return_label=False, sample_num=sp_num
        )

        cls.train_loader = fluid.io.DataLoader(
            cls.train_dataset, places=cls.device, batch_size=64
        )
        cls.val_loader = fluid.io.DataLoader(
            cls.val_dataset, places=cls.device, batch_size=64
        )
        cls.test_loader = fluid.io.DataLoader(
            cls.test_dataset, places=cls.device, batch_size=64
        )
=======
        cls.test_dataset = MnistDataset(mode='test',
                                        return_label=False,
                                        sample_num=sp_num)

        cls.train_loader = fluid.io.DataLoader(cls.train_dataset,
                                               places=cls.device,
                                               batch_size=64)
        cls.val_loader = fluid.io.DataLoader(cls.val_dataset,
                                             places=cls.device,
                                             batch_size=64)
        cls.test_loader = fluid.io.DataLoader(cls.test_dataset,
                                              places=cls.device,
                                              batch_size=64)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        seed = 333
        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)

        dy_lenet = LeNetDygraph()
        cls.init_param = dy_lenet.state_dict()
        dynamic_train(dy_lenet, cls.train_loader)

        cls.acc1 = dynamic_evaluate(dy_lenet, cls.val_loader)

        cls.inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
        cls.labels = [InputSpec([None, 1], 'int64', 'label')]

        cls.save_dir = os.path.join(tempfile.mkdtemp(), '.cache_test_model')
        if not os.path.exists(cls.save_dir):
            os.makedirs(cls.save_dir)
        cls.weight_path = os.path.join(cls.save_dir, 'lenet')
<<<<<<< HEAD
        paddle.save(dy_lenet.state_dict(), cls.weight_path + '.pdparams')
=======
        fluid.dygraph.save_dygraph(dy_lenet.state_dict(), cls.weight_path)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        fluid.disable_dygraph()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.save_dir)

    def test_fit_dygraph(self):
        self.fit(True)

    def test_fit_static(self):
        self.fit(False)

    def test_fit_dynamic_with_tuple_input(self):
        self.fit_with_tuple_input(True)

    def test_fit_static_with_tuple_input(self):
        self.fit_with_tuple_input(False)

    def test_fit_dynamic_with_rank(self):
        self.fit(True, 2, 0)

    def test_fit_static_with_rank(self):
        self.fit(False, 2, 0)

    def test_fit_dynamic_with_num_iters(self):
        self.fit(True, num_iters=1)

    def test_fit_static_with_num_iters(self):
        self.fit(False, num_iters=1)

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

    def fit(self, dynamic, num_replicas=None, rank=None, num_iters=None):
        fluid.enable_dygraph(self.device) if dynamic else None
        seed = 333
        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)

        net = LeNet()
<<<<<<< HEAD
        optim_new = fluid.optimizer.Adam(
            learning_rate=0.001, parameter_list=net.parameters()
        )
        model = Model(net, inputs=self.inputs, labels=self.labels)
        model.prepare(
            optim_new,
            loss=CrossEntropyLoss(reduction="sum"),
            metrics=Accuracy(),
        )
=======
        optim_new = fluid.optimizer.Adam(learning_rate=0.001,
                                         parameter_list=net.parameters())
        model = Model(net, inputs=self.inputs, labels=self.labels)
        model.prepare(optim_new,
                      loss=CrossEntropyLoss(reduction="sum"),
                      metrics=Accuracy())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        model.fit(self.train_dataset, batch_size=64, shuffle=False)

        result = model.evaluate(self.val_dataset, batch_size=64)
        np.testing.assert_allclose(result['acc'], self.acc1)

<<<<<<< HEAD
        model.fit(
            self.train_dataset,
            batch_size=64,
            shuffle=False,
            num_iters=num_iters,
        )

        result = model.evaluate(
            self.val_dataset, batch_size=64, num_iters=num_iters
        )

        model.fit(self.train_dataset, batch_size=(64, 64), shuffle=False)

        train_sampler = DistributedBatchSampler(
            self.train_dataset,
            batch_size=64,
            shuffle=False,
            num_replicas=num_replicas,
            rank=rank,
        )
        val_sampler = DistributedBatchSampler(
            self.val_dataset,
            batch_size=64,
            shuffle=False,
            num_replicas=num_replicas,
            rank=rank,
        )

        train_loader = fluid.io.DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            places=self.device,
            return_list=True,
        )

        val_loader = fluid.io.DataLoader(
            self.val_dataset,
            batch_sampler=val_sampler,
            places=self.device,
            return_list=True,
        )
=======
        model.fit(self.train_dataset,
                  batch_size=64,
                  shuffle=False,
                  num_iters=num_iters)

        result = model.evaluate(self.val_dataset,
                                batch_size=64,
                                num_iters=num_iters)

        train_sampler = DistributedBatchSampler(self.train_dataset,
                                                batch_size=64,
                                                shuffle=False,
                                                num_replicas=num_replicas,
                                                rank=rank)
        val_sampler = DistributedBatchSampler(self.val_dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              num_replicas=num_replicas,
                                              rank=rank)

        train_loader = fluid.io.DataLoader(self.train_dataset,
                                           batch_sampler=train_sampler,
                                           places=self.device,
                                           return_list=True)

        val_loader = fluid.io.DataLoader(self.val_dataset,
                                         batch_sampler=val_sampler,
                                         places=self.device,
                                         return_list=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        model.fit(train_loader, val_loader)
        fluid.disable_dygraph() if dynamic else None

    def fit_with_tuple_input(self, dynamic, num_replicas=None, rank=None):
        fluid.enable_dygraph(self.device) if dynamic else None
        seed = 333
        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)

        net = LeNet()
<<<<<<< HEAD
        optim_new = fluid.optimizer.Adam(
            learning_rate=0.001, parameter_list=net.parameters()
        )
        model = Model(net, inputs=tuple(self.inputs), labels=tuple(self.labels))
        model.prepare(
            optim_new,
            loss=CrossEntropyLoss(reduction="sum"),
            metrics=Accuracy(),
        )
=======
        optim_new = fluid.optimizer.Adam(learning_rate=0.001,
                                         parameter_list=net.parameters())
        model = Model(net, inputs=tuple(self.inputs), labels=tuple(self.labels))
        model.prepare(optim_new,
                      loss=CrossEntropyLoss(reduction="sum"),
                      metrics=Accuracy())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        model.fit(self.train_dataset, batch_size=64, shuffle=False)

        result = model.evaluate(self.val_dataset, batch_size=64)
        np.testing.assert_allclose(result['acc'], self.acc1)

<<<<<<< HEAD
        train_sampler = DistributedBatchSampler(
            self.train_dataset,
            batch_size=64,
            shuffle=False,
            num_replicas=num_replicas,
            rank=rank,
        )
        val_sampler = DistributedBatchSampler(
            self.val_dataset,
            batch_size=64,
            shuffle=False,
            num_replicas=num_replicas,
            rank=rank,
        )

        train_loader = fluid.io.DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            places=self.device,
            return_list=True,
        )

        val_loader = fluid.io.DataLoader(
            self.val_dataset,
            batch_sampler=val_sampler,
            places=self.device,
            return_list=True,
        )
=======
        train_sampler = DistributedBatchSampler(self.train_dataset,
                                                batch_size=64,
                                                shuffle=False,
                                                num_replicas=num_replicas,
                                                rank=rank)
        val_sampler = DistributedBatchSampler(self.val_dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              num_replicas=num_replicas,
                                              rank=rank)

        train_loader = fluid.io.DataLoader(self.train_dataset,
                                           batch_sampler=train_sampler,
                                           places=self.device,
                                           return_list=True)

        val_loader = fluid.io.DataLoader(self.val_dataset,
                                         batch_sampler=val_sampler,
                                         places=self.device,
                                         return_list=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        model.fit(train_loader, val_loader)
        fluid.disable_dygraph() if dynamic else None

    def evaluate(self, dynamic):
        fluid.enable_dygraph(self.device) if dynamic else None
        model = Model(LeNet(), self.inputs, self.labels)
        model.prepare(metrics=Accuracy())
        model.load(self.weight_path)
        result = model.evaluate(self.val_dataset, batch_size=64)
        np.testing.assert_allclose(result['acc'], self.acc1)

<<<<<<< HEAD
        sampler = DistributedBatchSampler(
            self.val_dataset, batch_size=64, shuffle=False
        )

        val_loader = fluid.io.DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            places=self.device,
            return_list=True,
        )
=======
        sampler = DistributedBatchSampler(self.val_dataset,
                                          batch_size=64,
                                          shuffle=False)

        val_loader = fluid.io.DataLoader(self.val_dataset,
                                         batch_sampler=sampler,
                                         places=self.device,
                                         return_list=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        model.evaluate(val_loader)

        fluid.disable_dygraph() if dynamic else None

    def predict(self, dynamic):
        fluid.enable_dygraph(self.device) if dynamic else None
        model = Model(LeNet(), self.inputs)
        model.prepare()
        model.load(self.weight_path)
<<<<<<< HEAD
        output = model.predict(
            self.test_dataset, batch_size=64, stack_outputs=True
        )
=======
        output = model.predict(self.test_dataset,
                               batch_size=64,
                               stack_outputs=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        np.testing.assert_equal(output[0].shape[0], len(self.test_dataset))

        acc = compute_acc(output[0], self.val_dataset.labels)
        np.testing.assert_allclose(acc, self.acc1)

<<<<<<< HEAD
        sampler = DistributedBatchSampler(
            self.test_dataset, batch_size=64, shuffle=False
        )

        test_loader = fluid.io.DataLoader(
            self.test_dataset,
            batch_sampler=sampler,
            places=self.device,
            return_list=True,
        )
=======
        sampler = DistributedBatchSampler(self.test_dataset,
                                          batch_size=64,
                                          shuffle=False)

        test_loader = fluid.io.DataLoader(self.test_dataset,
                                          batch_sampler=sampler,
                                          places=self.device,
                                          return_list=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        model.evaluate(test_loader)

        fluid.disable_dygraph() if dynamic else None

    def test_predict_without_inputs(self):
        fluid.enable_dygraph(self.device)
        model = Model(LeNet())
        model.prepare()
        model.load(self.weight_path)
        model._inputs = None
<<<<<<< HEAD
        output = model.predict(
            self.test_dataset, batch_size=64, stack_outputs=True
        )
=======
        output = model.predict(self.test_dataset,
                               batch_size=64,
                               stack_outputs=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        np.testing.assert_equal(output[0].shape[0], len(self.test_dataset))
        fluid.disable_dygraph()

    def test_summary_gpu(self):
        paddle.disable_static(self.device)
        rnn = paddle.nn.LSTM(16, 32, 2)
<<<<<<< HEAD
        params_info = paddle.summary(
            rnn, [(-1, 23, 16), ((2, None, 32), (2, -1, 32))]
        )


class MyModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
=======
        params_info = paddle.summary(rnn, [(-1, 23, 16),
                                           ((2, None, 32), (2, -1, 32))])


class MyModel(paddle.nn.Layer):

    def __init__(self):
        super(MyModel, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._fc = Linear(20, 10)

    def forward(self, x):
        y = self._fc(x)
        return y


class MyDataset(Dataset):
<<<<<<< HEAD
    def __getitem__(self, idx):
        return np.random.random(size=(20,)).astype(
            np.float32
        ), np.random.randint(0, 10, size=(1,)).astype(np.int64)
=======

    def __getitem__(self, idx):
        return np.random.random(size=(20,)).astype(np.float32), \
               np.random.randint(0, 10, size=(1,)).astype(np.int64)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def __len__(self):
        return 40


class TestModelFunction(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_seed(self, seed=1024):
        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)

    def test_train_batch(self, dynamic=True):
        dim = 20
        data = np.random.random(size=(4, dim)).astype(np.float32)
        label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)

        def get_expect():
            fluid.enable_dygraph(fluid.CPUPlace())
            self.set_seed()
            m = MyModel()
<<<<<<< HEAD
            optim = fluid.optimizer.SGD(
                learning_rate=0.001, parameter_list=m.parameters()
            )
            m.train()
            output = m(to_tensor(data))
            loss = CrossEntropyLoss(reduction='sum')(output, to_tensor(label))
            avg_loss = paddle.sum(loss)
=======
            optim = fluid.optimizer.SGD(learning_rate=0.001,
                                        parameter_list=m.parameters())
            m.train()
            output = m(to_tensor(data))
            loss = CrossEntropyLoss(reduction='sum')(output, to_tensor(label))
            avg_loss = fluid.layers.reduce_sum(loss)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            avg_loss.backward()
            optim.minimize(avg_loss)
            m.clear_gradients()
            fluid.disable_dygraph()
            return avg_loss.numpy()

        ref = get_expect()
        for dynamic in [True, False]:
            device = paddle.set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            self.set_seed()

            net = MyModel()
<<<<<<< HEAD
            optim2 = fluid.optimizer.SGD(
                learning_rate=0.001, parameter_list=net.parameters()
            )
=======
            optim2 = fluid.optimizer.SGD(learning_rate=0.001,
                                         parameter_list=net.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            inputs = [InputSpec([None, dim], 'float32', 'x')]
            labels = [InputSpec([None, 1], 'int64', 'label')]
            model = Model(net, inputs, labels)
            model.prepare(optim2, loss=CrossEntropyLoss(reduction="sum"))
<<<<<<< HEAD
            (loss,) = model.train_batch([data], [label])
=======
            loss, = model.train_batch([data], [label])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.testing.assert_allclose(loss.flatten(), ref.flatten())
            fluid.disable_dygraph() if dynamic else None

    def test_test_batch(self):
        dim = 20
        data = np.random.random(size=(4, dim)).astype(np.float32)

        def get_expect():
            fluid.enable_dygraph(fluid.CPUPlace())
            self.set_seed()
            m = MyModel()
            m.eval()
            output = m(to_tensor(data))
            fluid.disable_dygraph()
            return output.numpy()

        ref = get_expect()
        for dynamic in [True, False]:
            device = paddle.set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            self.set_seed()
            net = MyModel()
            inputs = [InputSpec([None, dim], 'float32', 'x')]
            model = Model(net, inputs)
            model.prepare()
<<<<<<< HEAD
            (out,) = model.predict_batch([data])
=======
            out, = model.predict_batch([data])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            np.testing.assert_allclose(out, ref, rtol=1e-6)
            fluid.disable_dygraph() if dynamic else None

    def test_save_load(self):
        path = os.path.join(tempfile.mkdtemp(), '.cache_test_save_load')
        if not os.path.exists(path):
            os.makedirs(path)
        for dynamic in [True, False]:
            device = paddle.set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            net = MyModel()
            inputs = [InputSpec([None, 20], 'float32', 'x')]
            labels = [InputSpec([None, 1], 'int64', 'label')]
<<<<<<< HEAD
            optim = fluid.optimizer.SGD(
                learning_rate=0.001, parameter_list=net.parameters()
            )
            model = Model(net, inputs, labels)
            model.prepare(
                optimizer=optim, loss=CrossEntropyLoss(reduction="sum")
            )
=======
            optim = fluid.optimizer.SGD(learning_rate=0.001,
                                        parameter_list=net.parameters())
            model = Model(net, inputs, labels)
            model.prepare(optimizer=optim,
                          loss=CrossEntropyLoss(reduction="sum"))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            model.save(path)
            model.load(path)
            fluid.disable_dygraph() if dynamic else None
        shutil.rmtree(path)

    def test_dynamic_load(self):
        mnist_data = MnistDataset(mode='train')

        path = os.path.join(tempfile.mkdtemp(), '.cache_dynamic_load')
        if not os.path.exists(path):
            os.makedirs(path)

        for new_optimizer in [True, False]:
            paddle.disable_static()
            net = LeNet()
            inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
            labels = [InputSpec([None, 1], 'int64', 'label')]
            if new_optimizer:
<<<<<<< HEAD
                optim = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=net.parameters()
                )
            else:
                optim = fluid.optimizer.Adam(
                    learning_rate=0.001, parameter_list=net.parameters()
                )
            model = Model(net, inputs, labels)
            model.prepare(
                optimizer=optim, loss=CrossEntropyLoss(reduction="sum")
            )
=======
                optim = paddle.optimizer.Adam(learning_rate=0.001,
                                              parameters=net.parameters())
            else:
                optim = fluid.optimizer.Adam(learning_rate=0.001,
                                             parameter_list=net.parameters())
            model = Model(net, inputs, labels)
            model.prepare(optimizer=optim,
                          loss=CrossEntropyLoss(reduction="sum"))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            model.fit(mnist_data, batch_size=64, verbose=0)
            model.save(path)
            model.load(path)
            paddle.enable_static()
        shutil.rmtree(path)

    def test_dynamic_save_static_load(self):
<<<<<<< HEAD
        path = os.path.join(
            tempfile.mkdtemp(), '.cache_dynamic_save_static_load'
        )
=======
        path = os.path.join(tempfile.mkdtemp(),
                            '.cache_dynamic_save_static_load')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if not os.path.exists(path):
            os.makedirs(path)
        # dynamic saving
        device = paddle.set_device('cpu')
        fluid.enable_dygraph(device)
        model = Model(MyModel())
<<<<<<< HEAD
        optim = fluid.optimizer.SGD(
            learning_rate=0.001, parameter_list=model.parameters()
        )
=======
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=model.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
        model.save(path)
        fluid.disable_dygraph()

        inputs = [InputSpec([None, 20], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        model = Model(MyModel(), inputs, labels)
<<<<<<< HEAD
        optim = fluid.optimizer.SGD(
            learning_rate=0.001, parameter_list=model.parameters()
        )
=======
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=model.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
        model.load(path)
        shutil.rmtree(path)

    def test_static_save_dynamic_load(self):
<<<<<<< HEAD
        path = os.path.join(
            tempfile.mkdtemp(), '.cache_test_static_save_dynamic_load'
        )
=======
        path = os.path.join(tempfile.mkdtemp(),
                            '.cache_test_static_save_dynamic_load')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if not os.path.exists(path):
            os.makedirs(path)
        net = MyModel()
        inputs = [InputSpec([None, 20], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
<<<<<<< HEAD
        optim = fluid.optimizer.SGD(
            learning_rate=0.001, parameter_list=net.parameters()
        )
=======
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=net.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        model = Model(net, inputs, labels)
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
        model.save(path)

        device = paddle.set_device('cpu')
<<<<<<< HEAD
        fluid.enable_dygraph(device)  # if dynamic else None
=======
        fluid.enable_dygraph(device)  #if dynamic else None
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        net = MyModel()
        inputs = [InputSpec([None, 20], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
<<<<<<< HEAD
        optim = fluid.optimizer.SGD(
            learning_rate=0.001, parameter_list=net.parameters()
        )
=======
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=net.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        model = Model(net, inputs, labels)
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
        model.load(path)
        shutil.rmtree(path)
        fluid.disable_dygraph()

    def test_parameters(self):
        for dynamic in [True, False]:
            device = paddle.set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            net = MyModel()
            inputs = [InputSpec([None, 20], 'float32', 'x')]
            model = Model(net, inputs)
            model.prepare()
            params = model.parameters()
            self.assertTrue(params[0].shape[0] == 20)
            self.assertTrue(params[0].shape[1] == 10)
            fluid.disable_dygraph() if dynamic else None

    def test_summary(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def _get_param_from_state_dict(state_dict):
            params = 0
            for k, v in state_dict.items():
                params += np.prod(v.numpy().shape)
            return params

        for dynamic in [True, False]:
            device = paddle.set_device('cpu')
            fluid.enable_dygraph(device) if dynamic else None
            net = MyModel()
            inputs = [InputSpec([None, 20], 'float32', 'x')]
            model = Model(net, inputs)
            model.prepare()
            params_info = model.summary()
            gt_params = _get_param_from_state_dict(net.state_dict())

            np.testing.assert_allclose(params_info['total_params'], gt_params)
            print(params_info)

            model.summary(input_size=(20))
            model.summary(input_size=[(20)])
            model.summary(input_size=(20), dtype='float32')

    def test_summary_non_tensor(self):
        paddle.summary(ModelOutter(), input_size=(-1, 3))

    def test_summary_nlp(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def _get_param_from_state_dict(state_dict):
            params = 0
            for k, v in state_dict.items():
                params += np.prod(v.numpy().shape)
            return params

<<<<<<< HEAD
        nlp_net = paddle.nn.GRU(
            input_size=2, hidden_size=3, num_layers=3, direction="bidirectional"
        )
        paddle.summary(nlp_net, (1, 1, 2))

        rnn = paddle.nn.LSTM(16, 32, 2)
        params_info = paddle.summary(
            rnn, [(-1, 23, 16), ((2, None, 32), (2, -1, 32))]
        )
=======
        nlp_net = paddle.nn.GRU(input_size=2,
                                hidden_size=3,
                                num_layers=3,
                                direction="bidirectional")
        paddle.summary(nlp_net, (1, 1, 2))

        rnn = paddle.nn.LSTM(16, 32, 2)
        params_info = paddle.summary(rnn, [(-1, 23, 16),
                                           ((2, None, 32), (2, -1, 32))])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        gt_params = _get_param_from_state_dict(rnn.state_dict())
        np.testing.assert_allclose(params_info['total_params'], gt_params / 2.0)

        rnn = paddle.nn.GRU(16, 32, 2, direction='bidirectional')
        params_info = paddle.summary(rnn, (4, 23, 16))
        gt_params = _get_param_from_state_dict(rnn.state_dict())
        np.testing.assert_allclose(params_info['total_params'], gt_params / 2.0)

        rnn = paddle.nn.SimpleRNN(16, 32, 2, direction='bidirectional')
        params_info = paddle.summary(rnn, (4, 23, 16))
        gt_params = _get_param_from_state_dict(rnn.state_dict())
        np.testing.assert_allclose(params_info['total_params'], gt_params / 2.0)

    def test_summary_input(self):
        paddle.enable_static()
        mymodel = MyModel()
        input_data = paddle.rand([1, 20])
        paddle.summary(mymodel, input=input_data)
        paddle.disable_static()

        rnn = paddle.nn.SimpleRNN(16, 32, 2, direction='bidirectional')
        input_data = paddle.rand([4, 23, 16])
        paddle.summary(rnn, input=input_data)

        lenet_List_input = LeNetListInput()
        input_data = [paddle.rand([1, 1, 28, 28]), paddle.rand([1, 400])]
        paddle.summary(lenet_List_input, input=input_data)

        lenet_dict_input = LeNetDictInput()
        input_data = {
            'x1': paddle.rand([1, 1, 28, 28]),
<<<<<<< HEAD
            'x2': paddle.rand([1, 400]),
=======
            'x2': paddle.rand([1, 400])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        paddle.summary(lenet_dict_input, input=input_data)

    def test_summary_dtype(self):
        input_shape = (3, 1)
        net = paddle.nn.Embedding(10, 3, sparse=True)
        paddle.summary(net, input_shape, dtypes='int64')

    def test_summary_error(self):
        with self.assertRaises(TypeError):
            nlp_net = paddle.nn.GRU(input_size=2, hidden_size=3, num_layers=3)
            paddle.summary(nlp_net, (1, 1, '2'))

        with self.assertRaises(ValueError):
            nlp_net = paddle.nn.GRU(input_size=2, hidden_size=3, num_layers=3)
            paddle.summary(nlp_net, (-1, -1))

        paddle.disable_static()
        nlp_net = paddle.nn.GRU(input_size=2, hidden_size=3, num_layers=3)
        paddle.summary(nlp_net, (1, 1, 2))

    def test_static_flops(self):
        if paddle.fluid.framework._in_eager_without_dygraph_check():
            return
        paddle.disable_static()
        net = models.__dict__['mobilenet_v2'](pretrained=False)
        inputs = paddle.randn([1, 3, 224, 224])
        static_program = jit._trace(net, inputs=[inputs])[1]
        paddle.flops(static_program, [1, 3, 224, 224], print_detail=True)

    def test_dynamic_flops(self):
        net = models.__dict__['mobilenet_v2'](pretrained=False)

        def customize_dropout(m, x, y):
            m.total_ops += 0

<<<<<<< HEAD
        paddle.flops(
            net,
            [1, 3, 224, 224],
            custom_ops={paddle.nn.Dropout: customize_dropout},
            print_detail=True,
        )

    def test_dynamic_flops_with_multiple_outputs(self):
        net = paddle.nn.MaxPool2D(
            kernel_size=2, stride=2, padding=0, return_mask=True
        )
=======
        paddle.flops(net, [1, 3, 224, 224],
                     custom_ops={paddle.nn.Dropout: customize_dropout},
                     print_detail=True)

    def test_dynamic_flops_with_multiple_outputs(self):
        net = paddle.nn.MaxPool2D(kernel_size=2,
                                  stride=2,
                                  padding=0,
                                  return_mask=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def customize_dropout(m, x, y):
            m.total_ops += 0

<<<<<<< HEAD
        paddle.flops(
            net,
            [1, 2, 32, 32],
            custom_ops={paddle.nn.Dropout: customize_dropout},
            print_detail=True,
        )
=======
        paddle.flops(net, [1, 2, 32, 32],
                     custom_ops={paddle.nn.Dropout: customize_dropout},
                     print_detail=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_export_deploy_model(self):
        self.set_seed()
        np.random.seed(201)

<<<<<<< HEAD
        save_dir = os.path.join(
            tempfile.mkdtemp(), '.cache_test_export_deploy_model'
        )
=======
        save_dir = os.path.join(tempfile.mkdtemp(),
                                '.cache_test_export_deploy_model')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for dynamic in [True, False]:
            paddle.disable_static() if dynamic else None
<<<<<<< HEAD
            paddle.jit.enable_to_static(False) if not dynamic else None

=======
            prog_translator = ProgramTranslator()
            prog_translator.enable(False) if not dynamic else None
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            net = LeNet()
            inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
            model = Model(net, inputs)
            model.prepare()

<<<<<<< HEAD
            tensor_img = np.array(
                np.random.random((1, 1, 28, 28)), dtype=np.float32
            )
=======
            tensor_img = np.array(np.random.random((1, 1, 28, 28)),
                                  dtype=np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            model.save(save_dir, training=False)
            ori_results = model.predict_batch(tensor_img)
            fluid.disable_dygraph() if dynamic else None

<<<<<<< HEAD
            place = (
                fluid.CPUPlace()
                if not fluid.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
            new_scope = fluid.Scope()
            with fluid.scope_guard(new_scope):
                exe = fluid.Executor(place)
                [
                    inference_program,
                    feed_target_names,
                    fetch_targets,
                ] = paddle.static.io.load_inference_model(
                    path_prefix=save_dir, executor=exe
                )
                results = exe.run(
                    inference_program,
                    feed={feed_target_names[0]: tensor_img},
                    fetch_list=fetch_targets,
                )
                np.testing.assert_allclose(
                    results, ori_results, rtol=1e-5, atol=1e-6
                )
=======
            place = fluid.CPUPlace(
            ) if not fluid.is_compiled_with_cuda() else fluid.CUDAPlace(0)
            new_scope = fluid.Scope()
            with fluid.scope_guard(new_scope):
                exe = fluid.Executor(place)
                [inference_program, feed_target_names,
                 fetch_targets] = (paddle.static.io.load_inference_model(
                     path_prefix=save_dir, executor=exe))
                results = exe.run(inference_program,
                                  feed={feed_target_names[0]: tensor_img},
                                  fetch_list=fetch_targets)
                np.testing.assert_allclose(results,
                                           ori_results,
                                           rtol=1e-5,
                                           atol=1e-6)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            paddle.enable_static()

        shutil.rmtree(save_dir)

    def test_dygraph_export_deploy_model_about_inputs(self):
        self.set_seed()
        np.random.seed(201)
        mnist_data = MnistDataset(mode='train')
        paddle.disable_static()
        # without inputs
<<<<<<< HEAD
        save_dir = os.path.join(
            tempfile.mkdtemp(), '.cache_test_dygraph_export_deploy'
        )
=======
        save_dir = os.path.join(tempfile.mkdtemp(),
                                '.cache_test_dygraph_export_deploy')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for initial in ["fit", "train_batch", "eval_batch", "predict_batch"]:
            net = LeNet()
            model = Model(net)
<<<<<<< HEAD
            optim = fluid.optimizer.Adam(
                learning_rate=0.001, parameter_list=model.parameters()
            )
            model.prepare(
                optimizer=optim, loss=CrossEntropyLoss(reduction="sum")
            )
            if initial == "fit":
                model.fit(mnist_data, batch_size=64, verbose=0)
            else:
                img = np.array(
                    np.random.random((1, 1, 28, 28)), dtype=np.float32
                )
=======
            optim = fluid.optimizer.Adam(learning_rate=0.001,
                                         parameter_list=model.parameters())
            model.prepare(optimizer=optim,
                          loss=CrossEntropyLoss(reduction="sum"))
            if initial == "fit":
                model.fit(mnist_data, batch_size=64, verbose=0)
            else:
                img = np.array(np.random.random((1, 1, 28, 28)),
                               dtype=np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                label = np.array(np.random.rand(1, 1), dtype=np.int64)
                if initial == "train_batch":
                    model.train_batch([img], [label])
                elif initial == "eval_batch":
                    model.eval_batch([img], [label])
                else:
                    model.predict_batch([img])

            model.save(save_dir, training=False)
        shutil.rmtree(save_dir)
        # with inputs, and the type of inputs is InputSpec
<<<<<<< HEAD
        save_dir = os.path.join(
            tempfile.mkdtemp(), '.cache_test_dygraph_export_deploy_2'
        )
=======
        save_dir = os.path.join(tempfile.mkdtemp(),
                                '.cache_test_dygraph_export_deploy_2')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        net = LeNet()
        inputs = InputSpec([None, 1, 28, 28], 'float32', 'x')
        model = Model(net, inputs)
<<<<<<< HEAD
        optim = fluid.optimizer.Adam(
            learning_rate=0.001, parameter_list=model.parameters()
        )
=======
        optim = fluid.optimizer.Adam(learning_rate=0.001,
                                     parameter_list=model.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
        model.save(save_dir, training=False)
        shutil.rmtree(save_dir)

<<<<<<< HEAD
    def test_accumulate(
        self,
    ):
=======
    def test_accumulate(self, ):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        dim = 20
        data = np.random.random(size=(4, dim)).astype(np.float32)
        label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)
        net = MyModel()
<<<<<<< HEAD
        optim = fluid.optimizer.SGD(
            learning_rate=0.001, parameter_list=net.parameters()
        )
=======
        optim = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=net.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        inputs = [InputSpec([None, dim], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]

        for amp_cfg in [None, 'O1']:
            model = Model(net, inputs, labels)
<<<<<<< HEAD
            model.prepare(
                optim,
                loss=CrossEntropyLoss(reduction="sum"),
                amp_configs=amp_cfg,
            )
            losses, grads = [], []
            for stat in [False, False, True]:
                (loss,) = model.train_batch([data], [label], update=stat)
=======
            model.prepare(optim,
                          loss=CrossEntropyLoss(reduction="sum"),
                          amp_configs=amp_cfg)
            losses, grads = [], []
            for stat in [False, False, True]:
                loss, = model.train_batch([data], [label], update=stat)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                losses.append(loss)
                grads.append([p.grad.numpy() for p in net.parameters()])

            for grad1, grad2, grad3 in zip(*grads):
                np.testing.assert_almost_equal(grad1 * 2, grad2, decimal=4)
<<<<<<< HEAD
                np.testing.assert_almost_equal(
                    grad3, np.zeros_like(grad3), decimal=4
                )
=======
                np.testing.assert_almost_equal(grad3,
                                               np.zeros_like(grad3),
                                               decimal=4)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            np.testing.assert_almost_equal(losses[0], losses[1], decimal=4)
            np.testing.assert_almost_equal(losses[0], losses[2], decimal=4)


class TestModelWithLRScheduler(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_fit_by_step(self):
        base_lr = 1e-3
        boundaries = [5, 8]

        def make_optimizer(parameters=None):
            momentum = 0.9
            weight_decay = 5e-4
            values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
            learning_rate = paddle.optimizer.lr.PiecewiseDecay(
<<<<<<< HEAD
                boundaries=boundaries, values=values
            )
            learning_rate = paddle.optimizer.lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=4,
                start_lr=base_lr / 5.0,
                end_lr=base_lr,
                verbose=True,
            )
            optimizer = paddle.optimizer.Momentum(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                parameters=parameters,
            )
=======
                boundaries=boundaries, values=values)
            learning_rate = paddle.optimizer.lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=4,
                start_lr=base_lr / 5.,
                end_lr=base_lr,
                verbose=True)
            optimizer = paddle.optimizer.Momentum(learning_rate=learning_rate,
                                                  weight_decay=weight_decay,
                                                  momentum=momentum,
                                                  parameters=parameters)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return optimizer

        # dynamic test
        device = paddle.set_device('cpu')
        fluid.enable_dygraph(device)
        net = MyModel()
        inputs = [InputSpec([None, 20], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        optim = make_optimizer(net.parameters())
        model = Model(net, inputs, labels)
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))

        dataset = MyDataset()
        model.fit(dataset, dataset, batch_size=4, epochs=10, num_workers=0)

<<<<<<< HEAD
        np.testing.assert_allclose(
            model._optimizer._learning_rate.last_lr,
            base_lr * (0.1 ** len(boundaries)),
        )
=======
        np.testing.assert_allclose(model._optimizer._learning_rate.last_lr,
                                   base_lr * (0.1**len(boundaries)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # static test
        paddle.enable_static()

        net = MyModel()
        inputs = [InputSpec([None, 20], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        optim = make_optimizer(net.parameters())
        model = Model(net, inputs, labels)
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))

        dataset = MyDataset()
        model.fit(dataset, dataset, batch_size=4, epochs=10, num_workers=0)

<<<<<<< HEAD
        np.testing.assert_allclose(
            model._optimizer._learning_rate.last_lr,
            base_lr * (0.1 ** len(boundaries)),
        )
=======
        np.testing.assert_allclose(model._optimizer._learning_rate.last_lr,
                                   base_lr * (0.1**len(boundaries)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_fit_by_epoch(self):
        base_lr = 1e-3
        boundaries = [5, 8]
        epochs = 10
        wamup_epochs = 4

        def make_optimizer(parameters=None):
            momentum = 0.9
            weight_decay = 5e-4
            values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
            learning_rate = paddle.optimizer.lr.PiecewiseDecay(
<<<<<<< HEAD
                boundaries=boundaries, values=values
            )
            learning_rate = paddle.optimizer.lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=wamup_epochs,
                start_lr=base_lr / 5.0,
                end_lr=base_lr,
                verbose=True,
            )
            optimizer = paddle.optimizer.Momentum(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                parameters=parameters,
            )
=======
                boundaries=boundaries, values=values)
            learning_rate = paddle.optimizer.lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=wamup_epochs,
                start_lr=base_lr / 5.,
                end_lr=base_lr,
                verbose=True)
            optimizer = paddle.optimizer.Momentum(learning_rate=learning_rate,
                                                  weight_decay=weight_decay,
                                                  momentum=momentum,
                                                  parameters=parameters)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return optimizer

        # dynamic test
        device = paddle.set_device('cpu')
        fluid.enable_dygraph(device)
        net = MyModel()
        inputs = [InputSpec([None, 20], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        optim = make_optimizer(net.parameters())
        model = Model(net, inputs, labels)
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))

        dataset = MyDataset()

<<<<<<< HEAD
        lr_scheduler_callback = paddle.callbacks.LRScheduler(
            by_step=False, by_epoch=True
        )

        model.fit(
            dataset,
            dataset,
            batch_size=4,
            epochs=epochs,
            num_workers=0,
            callbacks=lr_scheduler_callback,
        )
=======
        lr_scheduler_callback = paddle.callbacks.LRScheduler(by_step=False,
                                                             by_epoch=True)

        model.fit(dataset,
                  dataset,
                  batch_size=4,
                  epochs=epochs,
                  num_workers=0,
                  callbacks=lr_scheduler_callback)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        cnt = 0
        for b in boundaries:
            if b + wamup_epochs <= epochs:
                cnt += 1

<<<<<<< HEAD
        np.testing.assert_allclose(
            model._optimizer._learning_rate.last_lr, base_lr * (0.1**cnt)
        )
=======
        np.testing.assert_allclose(model._optimizer._learning_rate.last_lr,
                                   base_lr * (0.1**cnt))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # static test
        paddle.enable_static()

        net = MyModel()
        inputs = [InputSpec([None, 20], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        optim = make_optimizer(net.parameters())
        model = Model(net, inputs, labels)
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))

        dataset = MyDataset()

<<<<<<< HEAD
        lr_scheduler_callback = paddle.callbacks.LRScheduler(
            by_step=False, by_epoch=True
        )

        model.fit(
            dataset,
            dataset,
            batch_size=4,
            epochs=epochs,
            num_workers=0,
            callbacks=lr_scheduler_callback,
        )
=======
        lr_scheduler_callback = paddle.callbacks.LRScheduler(by_step=False,
                                                             by_epoch=True)

        model.fit(dataset,
                  dataset,
                  batch_size=4,
                  epochs=epochs,
                  num_workers=0,
                  callbacks=lr_scheduler_callback)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        cnt = 0
        for b in boundaries:
            if b + wamup_epochs <= epochs:
                cnt += 1

<<<<<<< HEAD
        np.testing.assert_allclose(
            model._optimizer._learning_rate.last_lr, base_lr * (0.1**cnt)
        )


class TestRaiseError(unittest.TestCase):
=======
        np.testing.assert_allclose(model._optimizer._learning_rate.last_lr,
                                   base_lr * (0.1**cnt))


class TestRaiseError(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_input_without_name(self):
        net = MyModel()
        inputs = [InputSpec([None, 10], 'float32')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        with self.assertRaises(ValueError):
            model = Model(net, inputs, labels)

    def test_static_without_inputs(self):
        paddle.enable_static()
        net = MyModel()
        with self.assertRaises(TypeError):
            model = Model(net)

    def test_save_infer_model_without_inputs_and_run_in_dygraph(self):
        paddle.disable_static()
        net = MyModel()
        save_dir = os.path.join(tempfile.mkdtemp(), '.cache_test_save_infer')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with self.assertRaises(RuntimeError):
            model = Model(net)
            model.save(save_dir, training=False)
        paddle.enable_static()
        shutil.rmtree(save_dir)

    def test_save_infer_model_without_file_prefix(self):
        paddle.enable_static()
        net = LeNet()
        inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
        model = Model(net, inputs)
        model.prepare()
        path = ""
<<<<<<< HEAD
        tensor_img = np.array(
            np.random.random((1, 1, 28, 28)), dtype=np.float32
        )
=======
        tensor_img = np.array(np.random.random((1, 1, 28, 28)),
                              dtype=np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with self.assertRaises(ValueError):
            model.save(path, training=False)


if __name__ == '__main__':
    unittest.main()
