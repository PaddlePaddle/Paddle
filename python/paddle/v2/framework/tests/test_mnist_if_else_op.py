import paddle.v2.framework.layers as layers
from paddle.v2.framework.framework import Program
from paddle.v2.framework.executor import Executor
from paddle.v2.framework.optimizer import MomentumOptimizer
import paddle.v2.framework.core as core
import paddle.v2 as paddle
import unittest
import numpy as np


class TestMNISTIfElseOp(unittest.TestCase):
    def test_raw_api(self):
        kwargs = {'startup_program': Program(), 'main_program': Program()}
        image = layers.data(
            name='x', shape=[784], data_type='float32', **kwargs)

        label = layers.data(name='y', shape=[1], data_type='int64', **kwargs)

        limit = layers.fill_constant_batch_size_like(
            input=label,
            data_type=int(core.DataType.INT64),
            shape=[1],
            value=5.0,
            input_dim_idx=int(0),
            output_dim_idx=int(0),
            **kwargs)

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

                tensor_x = core.LoDTensor()
                tensor_x.set(x_data, place)

                tensor_y = core.LoDTensor()
                tensor_y.set(y_data, place)

                outs = map(np.array,
                           exe.run(kwargs['main_program'],
                                   feed={'x': tensor_x,
                                         'y': tensor_y},
                                   fetch_list=[avg_loss]))
                print outs[0]
                if outs[0] < 1.0:
                    return
        self.assertFalse(True)


if __name__ == '__main__':
    unittest.main()
