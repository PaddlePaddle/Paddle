#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import os
import paddle
import unittest
import paddle.fluid as fluid
from mobilenet import MobileNet
from paddle.fluid.contrib.slim.core import Compressor
from paddle.fluid.contrib.slim.graph import GraphWrapper


class TestReader(unittest.TestCase):
    """
    Test API of quantization strategy.
    """

    def set_train_reader(self, image, label, place):
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=128)
        return train_reader

    def set_val_reader(self, image, label, place):
        val_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=128)
        return val_reader

    def set_feed_list(self, image, label):
        return [('img', image.name), ('label', label.name)]

    def quan(self, config_file):
        if not fluid.core.is_compiled_with_cuda():
            return
        class_dim = 10
        image_shape = [1, 28, 28]

        train_program = fluid.Program()
        startup_program = fluid.Program()
        val_program = fluid.Program()

        with fluid.program_guard(train_program, startup_program):
            with fluid.unique_name.guard():
                image = fluid.layers.data(
                    name='image', shape=image_shape, dtype='float32')
                image.stop_gradient = False
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64')
                out = MobileNet(name='quan').net(input=image,
                                                 class_dim=class_dim)
                print("out: {}".format(out.name))
                acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
                cost = fluid.layers.cross_entropy(input=out, label=label)
                avg_cost = fluid.layers.mean(x=cost)
        optimizer = fluid.optimizer.Momentum(
            momentum=0.9,
            learning_rate=0.01,
            regularization=fluid.regularizer.L2Decay(4e-5))

        val_program = train_program.clone(for_test=False)

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(startup_program)

        val_reader = self.set_val_reader(image, label, place)

        val_feed_list = self.set_feed_list(image, label)
        val_fetch_list = [('acc_top1', acc_top1.name), ('acc_top5',
                                                        acc_top5.name)]

        train_reader = self.set_train_reader(image, label, place)
        train_feed_list = self.set_feed_list(image, label)
        train_fetch_list = [('loss', avg_cost.name)]

        com_pass = Compressor(
            place,
            fluid.global_scope(),
            train_program,
            train_reader=train_reader,
            train_feed_list=train_feed_list,
            train_fetch_list=train_fetch_list,
            eval_program=val_program,
            eval_reader=val_reader,
            eval_feed_list=val_feed_list,
            eval_fetch_list=val_fetch_list,
            train_optimizer=optimizer)
        com_pass.config(config_file)
        eval_graph = com_pass.run()


class TestReader1(TestReader):
    def set_train_reader(self, image, label, place):
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[image, label], capacity=16, iterable=True)
        loader.set_sample_generator(
            paddle.dataset.mnist.train(), batch_size=128, places=place)
        return loader

    def set_val_reader(self, image, label, place):
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[image, label], capacity=16, iterable=True)
        loader.set_sample_generator(
            paddle.dataset.mnist.test(), batch_size=128, places=place)
        return loader

    def test_compression(self):
        self.quan("./quantization/compress_2.yaml")


if __name__ == '__main__':
    unittest.main()
