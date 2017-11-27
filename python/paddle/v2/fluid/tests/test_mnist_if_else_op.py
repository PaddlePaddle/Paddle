import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.framework import Program
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.optimizer import MomentumOptimizer
import paddle.v2.fluid.core as core
import paddle.v2 as paddle
import unittest
import numpy as np


class TestMNISTIfElseOp(unittest.TestCase):
    def test_raw_api(self):
        kwargs = {'startup_program': Program(), 'main_program': Program()}
        image = layers.data(name='x', shape=[784], dtype='float32', **kwargs)

        label = layers.data(name='y', shape=[1], dtype='int64', **kwargs)

        limit = layers.fill_constant_batch_size_like(
            input=label, dtype='int64', shape=[1], value=5.0, **kwargs)

        cond = layers.less_than(x=label, y=limit, **kwargs)
        true_image, false_image = layers.split_lod_tensor(
            input=image, mask=cond, **kwargs)

        true_out = layers.create_tensor(dtype='float32', **kwargs)
        true_cond = layers.ConditionalBlock([true_image], **kwargs)

        with true_cond.block():
            hidden = layers.fc(input=true_image, size=100, act='tanh', **kwargs)
            prob = layers.fc(input=hidden, size=10, act='softmax', **kwargs)
            layers.assign(input=prob, output=true_out, **kwargs)

        false_out = layers.create_tensor(dtype='float32', **kwargs)
        false_cond = layers.ConditionalBlock([false_image], **kwargs)

        with false_cond.block():
            hidden = layers.fc(input=false_image,
                               size=200,
                               act='tanh',
                               **kwargs)
            prob = layers.fc(input=hidden, size=10, act='softmax', **kwargs)
            layers.assign(input=prob, output=false_out, **kwargs)

        prob = layers.merge_lod_tensor(
            in_true=true_out, in_false=false_out, mask=cond, x=image, **kwargs)
        loss = layers.cross_entropy(input=prob, label=label, **kwargs)
        avg_loss = layers.mean(x=loss, **kwargs)

        optimizer = MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        optimizer.minimize(avg_loss, kwargs['startup_program'])

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=200)

        place = core.CPUPlace()
        exe = Executor(place)

        exe.run(kwargs['startup_program'])
        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                x_data = np.array(map(lambda x: x[0], data)).astype("float32")
                y_data = np.array(map(lambda x: x[1], data)).astype("int64")
                y_data = np.expand_dims(y_data, axis=1)

                outs = exe.run(kwargs['main_program'],
                               feed={'x': x_data,
                                     'y': y_data},
                               fetch_list=[avg_loss])
                print outs[0]
                if outs[0] < 1.0:
                    return
        self.assertFalse(True)

    def test_ifelse(self):
        kwargs = {'startup_program': Program(), 'main_program': Program()}
        image = layers.data(name='x', shape=[784], dtype='float32', **kwargs)

        label = layers.data(name='y', shape=[1], dtype='int64', **kwargs)

        limit = layers.fill_constant_batch_size_like(
            input=label, dtype='int64', shape=[1], value=5.0, **kwargs)

        cond = layers.less_than(x=label, y=limit, **kwargs)

        ie = layers.IfElse(cond, **kwargs)

        with ie.true_block():
            true_image = ie.input(image)
            hidden = layers.fc(input=true_image, size=100, act='tanh', **kwargs)
            prob = layers.fc(input=hidden, size=10, act='softmax', **kwargs)
            ie.output(prob)

        with ie.false_block():
            false_image = ie.input(image)
            hidden = layers.fc(input=false_image,
                               size=200,
                               act='tanh',
                               **kwargs)
            prob = layers.fc(input=hidden, size=10, act='softmax', **kwargs)
            ie.output(prob)

        prob = ie()
        loss = layers.cross_entropy(input=prob[0], label=label, **kwargs)
        avg_loss = layers.mean(x=loss, **kwargs)

        optimizer = MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        optimizer.minimize(avg_loss, kwargs['startup_program'])
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=200)

        place = core.CPUPlace()
        exe = Executor(place)

        exe.run(kwargs['startup_program'])
        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                x_data = np.array(map(lambda x: x[0], data)).astype("float32")
                y_data = np.array(map(lambda x: x[1], data)).astype("int64")
                y_data = y_data.reshape((y_data.shape[0], 1))

                outs = exe.run(kwargs['main_program'],
                               feed={'x': x_data,
                                     'y': y_data},
                               fetch_list=[avg_loss])
                print outs[0]
                if outs[0] < 1.0:
                    return
        self.assertFalse(True)


if __name__ == '__main__':
    unittest.main()
