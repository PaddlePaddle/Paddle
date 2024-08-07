# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle
from paddle import base

paddle.enable_static()

dev_cnt = 2
if base.core.is_compiled_with_cuda():
    dev_cnt = base.core.get_cuda_device_count()
os.environ['CPU_NUM'] = str(dev_cnt)


def dummy_func_with_no_input():
    return np.array([0], dtype='float32')


def dummy_func_with_no_output(x):
    pass


def dummy_func_with_multi_input_output(x, y):
    return np.array(x), np.array(y)


def tanh(x):
    return np.tanh(x)


def tanh_grad(y, dy):
    return np.array(dy) * (1 - np.square(np.array(y)))


def cross_entropy(logits, labels):
    logits = np.array(logits)
    labels = np.array(labels)
    M = logits.shape[0]
    N = logits.shape[1]
    ret = np.ndarray([M, 1]).astype(logits.dtype)
    for idx in range(M):
        ret[idx][0] = -np.log(logits[idx][labels[idx][0]])
    return ret


def cross_entropy_grad(logits, labels, bwd_dout):
    logits = np.array(logits)
    labels = np.array(labels)
    bwd_dout = np.array(bwd_dout)
    M = logits.shape[0]
    N = logits.shape[1]
    dlogits = np.zeros([M, N]).astype(logits.dtype)
    for idx in range(M):
        dlogits[idx][labels[idx][0]] = (
            -bwd_dout[idx] / logits[idx][labels[idx][0]]
        )
    return dlogits, None


def simple_fc_net(img, label, use_py_func_op):
    hidden = img
    for idx in range(4):
        hidden = paddle.static.nn.fc(
            hidden,
            size=200,
            bias_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            ),
        )
        if not use_py_func_op:
            hidden = paddle.tanh(hidden)
        else:
            new_hidden = (
                base.default_main_program()
                .current_block()
                .create_var(
                    name=f'hidden_{idx}',
                    dtype='float32',
                    shape=hidden.shape,
                )
            )
            hidden = paddle.static.py_func(
                func=tanh,
                x=hidden,
                out=new_hidden,
                backward_func=tanh_grad,
                skip_vars_in_backward_input=hidden,
            )

    prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
    if not use_py_func_op:
        loss = paddle.nn.functional.cross_entropy(
            input=prediction, label=label, reduction='none', use_softmax=False
        )
    else:
        loss = (
            base.default_main_program()
            .current_block()
            .create_var(name='loss', dtype='float32', shape=[-1, 1])
        )
        loss = paddle.static.py_func(
            func=cross_entropy,
            x=[prediction, label],
            out=loss,
            backward_func=cross_entropy_grad,
            skip_vars_in_backward_input=loss,
        )

        dummy_var = (
            base.default_main_program()
            .current_block()
            .create_var(name='test_tmp_var', dtype='float32', shape=[1])
        )
        paddle.static.py_func(
            func=dummy_func_with_no_input, x=None, out=dummy_var
        )
        loss += dummy_var
        paddle.static.py_func(func=dummy_func_with_no_output, x=loss, out=None)

        loss_out = (
            base.default_main_program()
            .current_block()
            .create_var(dtype='float32', shape=[-1, 1])
        )
        dummy_var_out = (
            base.default_main_program()
            .current_block()
            .create_var(dtype='float32', shape=[1])
        )
        paddle.static.py_func(
            func=dummy_func_with_multi_input_output,
            x=(loss, dummy_var),
            out=(loss_out, dummy_var_out),
        )
        assert (
            loss == loss_out and dummy_var == dummy_var_out
        ), "py_func failed with multi input and output"

        paddle.static.py_func(
            func=dummy_func_with_multi_input_output,
            x=[loss, dummy_var],
            out=[loss_out, dummy_var_out],
        )
        assert (
            loss == loss_out and dummy_var == dummy_var_out
        ), "py_func failed with multi input and output"

    loss = paddle.mean(loss)
    return loss


def reader():
    for _ in range(dev_cnt * 100):
        yield np.random.random([784]), np.random.random_integers(
            size=[1], low=0, high=9
        )


def test_main(use_cuda, use_py_func_op):
    if use_cuda and not base.core.is_compiled_with_cuda():
        return None

    with base.program_guard(base.Program(), base.Program()):
        with base.scope_guard(base.core.Scope()):
            gen = paddle.seed(1)
            np.random.seed(1)
            img = paddle.static.data(
                name='image', shape=[-1, 784], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )
            loss = simple_fc_net(img, label, use_py_func_op)
            optimizer = paddle.optimizer.SGD(learning_rate=1e-3)
            optimizer.minimize(loss)

            place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
            feeder = base.DataFeeder(feed_list=[img, label], place=place)
            r = paddle.batch(reader, batch_size=10)

            exe = base.Executor(place)
            exe.run(base.default_startup_program())

            train_cp = base.default_main_program()
            fetch_list = [loss]

            ret = []
            for epoch_id in range(2):
                for d in r():
                    (L,) = exe.run(
                        train_cp, feed=feeder.feed(d), fetch_list=fetch_list
                    )
                    ret.append(L)
            return np.array(ret)


class TestPyFuncOpUseExecutor(unittest.TestCase):
    def test_loss_diff(self):
        for use_cuda in [True, False]:
            losses = []
            for use_py_func_op in [True, False]:
                L = test_main(use_cuda, use_py_func_op)
                if L is not None:
                    losses.append(L)

                for idx in range(len(losses) - 1):
                    max_diff = np.max(np.abs(losses[idx] - losses[0]))
                    self.assertAlmostEqual(max_diff, 0, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
