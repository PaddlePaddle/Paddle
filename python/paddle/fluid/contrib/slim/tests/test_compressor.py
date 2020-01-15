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
import os
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.contrib.slim.core import Compressor
from paddle.fluid.contrib.slim.graph import GraphWrapper


class TestCompressor(unittest.TestCase):
    def test_eval_func(self):
        class_dim = 10
        image_shape = [1, 28, 28]
        image = fluid.layers.data(
            name='image', shape=image_shape, dtype='float32')
        image.stop_gradient = False
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        out = fluid.layers.fc(input=image, size=class_dim)
        out = fluid.layers.softmax(out)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        val_program = fluid.default_main_program().clone(for_test=False)

        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        optimizer = fluid.optimizer.Momentum(
            momentum=0.9,
            learning_rate=0.01,
            regularization=fluid.regularizer.L2Decay(4e-5))

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        val_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=128)

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=128)
        train_feed_list = [('img', image.name), ('label', label.name)]
        train_fetch_list = [('loss', avg_cost.name)]
        eval_feed_list = [('img', image.name), ('label', label.name)]
        eval_fetch_list = [('acc_top1', acc_top1.name)]

        def eval_func(program, scope):
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(
                feed_list=[image.name, label.name],
                place=place,
                program=program)
            results = []
            for data in val_reader():
                result = exe.run(program=program,
                                 scope=scope,
                                 fetch_list=[acc_top1.name],
                                 feed=feeder.feed(data))
                results.append(np.array(result))
            result = np.mean(results)
            return result

        com_pass = Compressor(
            place,
            fluid.global_scope(),
            fluid.default_main_program(),
            train_reader=train_reader,
            train_feed_list=train_feed_list,
            train_fetch_list=train_fetch_list,
            eval_program=val_program,
            eval_feed_list=eval_feed_list,
            eval_fetch_list=eval_fetch_list,
            eval_func={"score": eval_func},
            prune_infer_model=[[image.name], [out.name]],
            train_optimizer=optimizer)
        com_pass.config('./configs/compress.yaml')
        com_pass.run()
        self.assertTrue('score' in com_pass.context.eval_results)
        self.assertTrue(float(com_pass.context.eval_results['score'][0]) > 0.9)
        self.assertTrue(os.path.exists("./checkpoints/0/eval_model/__model__"))
        self.assertTrue(
            os.path.exists("./checkpoints/0/eval_model/__model__.infer"))
        self.assertTrue(os.path.exists("./checkpoints/0/eval_model/__params__"))


if __name__ == '__main__':
    unittest.main()
