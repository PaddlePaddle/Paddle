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

import paddle
import unittest
import paddle.fluid as fluid
from filter_pruning.mobilenet import MobileNet
from paddle.fluid.contrib.slim.core import Compressor
from paddle.fluid.contrib.slim.graph import GraphWrapper


class TestFilterPruning(unittest.TestCase):
    def test_compression(self):
        """
        Model: mobilenet_v1
        data: mnist
        step1: Training one epoch
        step2: pruning flops
        step3: fine-tune one epoch
        step4: check top1_acc.
        """
        if not fluid.core.is_compiled_with_cuda():
            return
        class_dim = 10
        image_shape = [1, 28, 28]
        image = fluid.layers.data(
            name='image', shape=image_shape, dtype='float32')
        image.stop_gradient = False
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        out = MobileNet().net(input=image, class_dim=class_dim)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
        val_program = fluid.default_main_program().clone(for_test=False)

        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        optimizer = fluid.optimizer.Momentum(
            momentum=0.9,
            learning_rate=0.01,
            regularization=fluid.regularizer.L2Decay(4e-5))

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        val_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=128)

        val_feed_list = [('img', image.name), ('label', label.name)]
        val_fetch_list = [('acc_top1', acc_top1.name), ('acc_top5',
                                                        acc_top5.name)]

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=128)
        train_feed_list = [('img', image.name), ('label', label.name)]
        train_fetch_list = [('loss', avg_cost.name)]

        com_pass = Compressor(
            place,
            fluid.global_scope(),
            fluid.default_main_program(),
            train_reader=train_reader,
            train_feed_list=train_feed_list,
            train_fetch_list=train_fetch_list,
            eval_program=val_program,
            eval_reader=val_reader,
            eval_feed_list=val_feed_list,
            eval_fetch_list=val_fetch_list,
            train_optimizer=optimizer)
        com_pass.config('./filter_pruning/compress.yaml')
        eval_graph = com_pass.run()
        self.assertTrue(
            abs((com_pass.context.eval_results['acc_top1'][-1] - 0.969) / 0.969)
            < 0.02)


if __name__ == '__main__':
    unittest.main()
