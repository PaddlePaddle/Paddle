#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
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

import unittest
import random
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.slim.graph.executor import get_executor
from paddle.fluid.contrib.slim.graph import ImitationGraph
from paddle.fluid.contrib.slim.graph import load_inference_graph_model
from paddle.fluid.contrib.slim.quantization import StaticQuantizer
from paddle.fluid.contrib.slim.quantization import DynamicQuantizer


def conv_net(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    return avg_loss


class TestQuantizer(unittest.TestCase):
    def graph_quantize(self, use_cuda, seed, is_static=False):
        def build_program(main, startup, is_test):
            main.random_seed = seed
            startup.random_seed = seed
            with fluid.unique_name.guard():
                with fluid.program_guard(main, startup):
                    img = fluid.layers.data(
                        name='image', shape=[1, 28, 28], dtype='float32')
                    label = fluid.layers.data(
                        name='label', shape=[1], dtype='int64')
                    loss = conv_net(img, label)
                    if not is_test:
                        opt = fluid.optimizer.Adam(learning_rate=0.001)
                        opt.minimize(loss)
            return [img, label], loss

        random.seed(0)
        np.random.seed(0)

        main = fluid.Program()
        startup = fluid.Program()
        feeds, loss = build_program(main, startup, False)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        program_exe = fluid.Executor(place)
        program_exe.run(startup)

        graph = ImitationGraph(main)
        quantizer = StaticQuantizer() if is_static else DynamicQuantizer()
        quantizer.quantize(graph, program_exe, fluid.global_scope())

        iters = 5
        batch_size = 8
        class_num = 10

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)
        feeder = fluid.DataFeeder(feed_list=feeds, place=place)
        exe = get_executor(graph, place)

        for _ in range(iters):
            data = next(train_reader())
            _ = exe.run(graph=graph,
                        feed=feeder.feed(data),
                        fetches=[loss.name])
        save_path = 'static_quantize' if is_static else 'dynamic_quantize'
        save_path = save_path + '_gpu' if use_cuda else save_path + '_cpu'
        test_graph = quantizer.convert_model(
            graph,
            place,
            fluid.global_scope(),
            feeds,
            loss,
            dirname=save_path,
            exe=exe,
            target_device='mobile',
            save_as_int8=True)
        # Test whether the 8-bit parameter and model file can be loaded successfully.
        [infer, feed, fetch] = load_inference_graph_model(save_path, exe)
        # Check the loaded 8-bit weight.
        w_8bit = np.array(fluid.global_scope().find_var('conv2d_1.w_0.int8')
                          .get_tensor())

        w_freeze = np.array(fluid.global_scope().find_var('conv2d_1.w_0')
                            .get_tensor())
        self.assertEqual(w_8bit.dtype, np.int8)
        self.assertEqual(np.sum(w_8bit), np.sum(w_freeze))

    def test_graph_quantize_cuda_static(self):
        if fluid.core.is_compiled_with_cuda():
            with fluid.unique_name.guard():
                self.graph_quantize(True, seed=1, is_static=True)

    def test_graph_quantize_cuda_dynamic(self):
        if fluid.core.is_compiled_with_cuda():
            with fluid.unique_name.guard():
                self.graph_quantize(True, seed=2, is_static=False)

    def test_graph_quantize_cpu_static(self):
        with fluid.unique_name.guard():
            self.graph_quantize(False, seed=3, is_static=True)

    def test_graph_quantize_cpu_dynamic(self):
        with fluid.unique_name.guard():
            self.graph_quantize(False, seed=4, is_static=False)


if __name__ == '__main__':
    unittest.main()
