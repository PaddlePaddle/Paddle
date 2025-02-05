#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import functools
import sys
import unittest

sys.path.append("../../legacy_test")
import nets

import paddle
from paddle import base
from paddle.base import core

SEED = 1
DTYPE = "float32"
paddle.dataset.mnist.fetch()
paddle.enable_static()


def cnn_model(data):
    conv_pool_1 = nets.simple_img_conv_pool(
        input=data,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    conv_pool_2 = nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )

    # TODO(dzhwinter) : refine the initializer and random seed setting
    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [
        functools.reduce(lambda a, b: a * b, input_shape[1:], 1),
        SIZE,
    ]
    scale = (2.0 / (param_shape[0] ** 2 * SIZE)) ** 0.5

    predict = paddle.static.nn.fc(
        x=conv_pool_2,
        size=SIZE,
        activation="softmax",
        weight_attr=base.param_attr.ParamAttr(
            initializer=paddle.nn.initializer.Normal(loc=0.0, scale=scale)
        ),
    )
    return predict


def get_model(batch_size):
    # Input data
    images = paddle.static.data(
        name='pixel', shape=[-1, 1, 28, 28], dtype=DTYPE
    )
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')

    # Train program
    predict = cnn_model(images)
    cost = paddle.nn.functional.cross_entropy(
        input=predict, label=label, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(x=cost)

    # Evaluator
    batch_size_tensor = paddle.tensor.create_tensor(dtype='int64')
    batch_acc = paddle.static.accuracy(
        input=predict, label=label, total=batch_size_tensor
    )

    inference_program = base.default_main_program().clone()
    # Optimization
    opt = paddle.optimizer.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

    # Reader
    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=batch_size
    )
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=batch_size
    )
    opt.minimize(avg_cost)
    return (
        inference_program,
        avg_cost,
        train_reader,
        test_reader,
        batch_acc,
        predict,
    )


def operator_equal(a, b):
    if a.__str__() != b.__str__():
        raise ValueError("In operator_equal not equal\n")

    for k, v in a.__dict__.items():
        if isinstance(v, (base.framework.Program, base.framework.Block)):
            continue

        elif isinstance(v, core.OpDesc):
            continue

        elif isinstance(v, collections.OrderedDict):
            v0 = sorted(v.items(), key=lambda x: x[0])
            v1 = sorted(b.__dict__[k].items(), key=lambda x: x[0])

            if v0 != v1:
                raise ValueError(f"In operator_equal not equal:{k}\n")

        elif v != b.__dict__[k]:
            raise ValueError(f"In operator_equal not equal:{k}\n")

    return True


def block_equal(a, b):
    for k, v in a.__dict__.items():
        if isinstance(
            v, (core.ProgramDesc, base.framework.Program, core.BlockDesc)
        ):
            continue
        elif k == "ops":
            assert len(a.ops) == len(b.ops)
            for i in range(0, len(a.ops)):
                if not operator_equal(a.ops[i], b.ops[i]):
                    raise ValueError(f"In block_equal not equal:{k}\n")

        elif isinstance(v, collections.OrderedDict):
            for key, value in v.items():
                if str(value) != str(b.__dict__[k][key]):
                    raise ValueError(f"In block_equal not equal:{k}\n")

        elif v != b.__dict__[k]:
            raise ValueError(f"In block_equal not equal:{k}\n")

    return True


def program_equal(a, b):
    for k, v in a.__dict__.items():
        if isinstance(v, core.ProgramDesc):
            continue

        elif k == 'blocks':
            for i in range(0, len(a.blocks)):
                if not block_equal(a.blocks[i], b.blocks[i]):
                    raise ValueError(f"In operator_equal not equal:{k}\n")
                    return False
            assert len(a.blocks) == len(b.blocks)
        elif k == '_auto_checkpoint_name':
            continue
        elif v != b.__dict__[k]:
            raise ValueError(f"In program_equal not equal:{k}\n")

    return True


class TestCloneWithStopGradient(unittest.TestCase):
    def test_clone_with_stop_gradient(self):
        train_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(train_program, startup_program):
            img = paddle.static.data(name='image', shape=[-1, 784])
            hidden1 = paddle.static.nn.fc(x=img, size=200, activation='relu')
            hidden1.stop_gradient = True
            hidden2 = paddle.nn.functional.dropout(hidden1, p=0.5)
            loss = paddle.nn.functional.cross_entropy(
                input=paddle.static.nn.fc(
                    hidden2, size=10, activation='softmax'
                ),
                label=paddle.static.data(
                    name='label', shape=[-1, 1], dtype='int64'
                ),
                reduction='none',
                use_softmax=False,
            )
            avg_loss = paddle.mean(loss)
            test_program = train_program.clone(for_test=False)

        self.assertEqual(
            test_program.block(0).var(hidden1.name).stop_gradient, True
        )
        self.assertEqual(
            test_program.block(0).var(hidden2.name).stop_gradient, True
        )


class TestCloneWithStopGradientInSubBlock(unittest.TestCase):
    def test_clone_with_stop_gradient(self):
        train_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(train_program, startup_program):
            img = paddle.static.data(name='image', shape=[-1, 784])
            true = paddle.ones(shape=[1], dtype="float32")
            hidden1 = paddle.static.nn.fc(x=img, size=200, activation='relu')
            hidden1.stop_gradient = True

            cond = paddle.equal(true, true)

            def true_fn():
                hidden2 = paddle.nn.functional.dropout(hidden1, p=0.5)
                hidden2.stop_gradient = True
                return hidden2

            def false_fn():
                hidden2 = paddle.nn.functional.dropout(hidden1, p=0.6)
                return hidden2

            hidden2 = paddle.static.nn.cond(cond, true_fn, false_fn)

            loss = paddle.nn.functional.cross_entropy(
                input=paddle.static.nn.fc(
                    hidden2, size=10, activation='softmax'
                ),
                label=paddle.static.data(
                    name='label', shape=[-1, 1], dtype='int64'
                ),
                reduction='none',
                use_softmax=False,
            )
            avg_loss = paddle.mean(loss)
            test_program = train_program.clone(for_test=False)

        self.assertEqual(
            test_program.block(0).var(hidden1.name).stop_gradient, True
        )
        for var in test_program.block(1).vars.values():
            var2 = train_program.block(1).var(var.name)
            self.assertEqual(var.stop_gradient, var2.stop_gradient)
        for var in test_program.block(2).vars.values():
            var2 = train_program.block(2).var(var.name)
            self.assertEqual(var.stop_gradient, var2.stop_gradient)


class TestCloneWithRaise(unittest.TestCase):
    def test_clone_with_stop_gradient(self):
        train_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(train_program, startup_program):
            img = paddle.static.data(name='image', shape=[-1, 784])
            true = paddle.ones(shape=[1], dtype="float32")
            hidden1 = paddle.static.nn.fc(x=img, size=200, activation='relu')
            hidden1.stop_gradient = True

            cond = paddle.equal(true, true)

            def true_fn():
                hidden2 = paddle.nn.functional.dropout(hidden1, p=0.5)
                hidden2.stop_gradient = True
                return hidden2

            def false_fn():
                hidden2 = paddle.nn.functional.dropout(hidden1, p=0.6)
                return hidden2

            hidden2 = paddle.static.nn.cond(cond, true_fn, false_fn)
            loss = paddle.nn.functional.cross_entropy(
                input=paddle.static.nn.fc(
                    hidden2, size=10, activation='softmax'
                ),
                label=paddle.static.data(
                    name='label', shape=[-1, 1], dtype='int64'
                ),
                reduction='none',
                use_softmax=False,
            )
            avg_loss = paddle.mean(loss)
            test_program = train_program.clone(for_test=False)

        self.assertRaises(
            ValueError, train_program._copy_data_info_from, startup_program
        )
        self.assertRaises(
            TypeError,
            train_program._copy_data_info_from,
            startup_program.block(0),
        )


if __name__ == "__main__":
    unittest.main()
