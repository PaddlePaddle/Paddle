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

import numpy as np
import unittest
import paddle
import paddle.fluid as fluid


def linear_fc(num):
    data = fluid.layers.data(name='image', shape=[1, 32, 32], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = data
    for _ in xrange(num):
        hidden = fluid.layers.fc(hidden, size=128, act='relu')
    loss = fluid.layers.cross_entropy(input=hidden, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def residual_block(num):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    data = fluid.layers.data(name='image', shape=[1, 32, 32], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = data
    for _ in xrange(num):
        conv = conv_bn_layer(hidden, 16, 3, 1, 1, act=None, bias_attr=True)
        short = conv_bn_layer(hidden, 16, 1, 1, 0, act=None)
        hidden = fluid.layers.elementwise_add(x=conv, y=short, act='relu')
    fc = fluid.layers.fc(input=hidden, size=10)
    loss = fluid.layers.cross_entropy(input=fc, label=label)
    loss = fluid.layers.mean(loss)
    return loss


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


class TestQuantizeTranspiler(unittest.TestCase):
    def setUp(self):
        # since quant_op and dequant_op is not ready, use cos and sin for test
        self.quant_op_type = 'fake_quantize'
        self.dequant_op_type = 'fake_dequantize_max_abs'
        self.quantizable_op_and_inputs = {
            'conv2d': ['Input', 'Filter'],
            'depthwise_conv2d': ['Input', 'Filter'],
            'mul': ['X', 'Y']
        }
        self.quantizable_op_grad_and_inputs = {
            'conv2d_grad': ['Input', 'Filter'],
            'depthwise_conv2d_grad': ['Input', 'Filter'],
            'mul_grad': ['X', 'Y']
        }

    def check_program(self, program):
        quantized_ops = {}
        for block in program.blocks:
            for idx, op in enumerate(block.ops):
                # check forward
                if op.type in self.quantizable_op_and_inputs:
                    for i, arg_name in enumerate(op.input_arg_names):
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized'))
                        if arg_name not in quantized_ops:
                            self.assertEqual(block.ops[idx - 2 * i - 1].type,
                                             self.dequant_op_type)
                            self.assertEqual(block.ops[idx - 2 * i - 2].type,
                                             self.quant_op_type)
                            quantized_ops[arg_name] = block.ops[idx - 2 * i - 2]
                        else:
                            op_idx = block.ops.index(quantized_ops[arg_name])
                            self.assertLess(op_idx, idx)

                # check backward
                if op.type in self.quantizable_op_grad_and_inputs:
                    for pname in self.quantizable_op_grad_and_inputs[op.type]:
                        arg_name = op.input(pname)[0]
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized'))
                        self.assertTrue(arg_name in quantized_ops)

    def test_linear_fc(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = linear_fc(3)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
            t = fluid.QuantizeTranspiler()
            t.transpile(main)
            self.check_program(main)

    def test_residual_block(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = residual_block(2)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
            t = fluid.QuantizeTranspiler()
            t.transpile(main)
            self.check_program(main)

    def freeze_program(self, use_cuda):
        main = fluid.Program()
        startup = fluid.Program()
        quant_transpiler = fluid.QuantizeTranspiler()
        with fluid.program_guard(main, startup):
            img = fluid.layers.data(
                name='image', shape=[1, 28, 28], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            loss = conv_net(img, label)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
            quant_transpiler.transpile(main)

        test_program = main.clone()
        with fluid.program_guard(test_program):
            test_program = fluid.io.get_inference_program(loss)

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        iter = 5
        batch_size = 8
        class_num = 10
        exe.run(startup)

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)
        feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

        for _ in range(iter):
            data = train_reader().next()
            loss_v = exe.run(program=main,
                             feed=feeder.feed(data),
                             fetch_list=[loss])
        test_data = test_reader().next()

        f_var = fluid.framework.get_var('conv2d_1.tmp_0', test_program)
        w_var = fluid.framework.get_var('conv2d_1.w_0.quantized', test_program)
        # Testing during training
        test_loss, f_v, w_quant = exe.run(program=test_program,
                                          feed=feeder.feed(test_data),
                                          fetch_list=[loss, f_var, w_var])
        print("Test loss ", test_loss)

        # freeze program for inference, but the weight of fc/conv is still float type.
        quant_transpiler.freeze_program(test_program, place)
        fv2 = fluid.framework.get_var('conv2d_1.tmp_0.dequantized',
                                      test_program)
        test_loss, f_v = exe.run(program=test_program,
                                 feed=feeder.feed(test_data),
                                 fetch_list=[loss, fv2])
        print("Test loss ", test_loss)
        w_freeze = np.array(fluid.global_scope().find_var('conv2d_1.w_0')
                            .get_tensor())
        self.assertEqual(np.sum(w_freeze), np.sum(w_quant))

        # Convert parameter to 8-bit.
        quant_transpiler.convert_to_int8(test_program, place)
        # Save the 8-bit parameter and model file.
        fluid.io.save_inference_model('model_8bit', ['image', 'label'], [loss],
                                      exe, test_program)
        # Test whether the 8-bit parameter and model file can be loaded successfully.
        [infer, feed, fetch] = fluid.io.load_inference_model('model_8bit', exe)
        # Check the loaded 8-bit weight.
        w_8bit = np.array(fluid.global_scope().find_var('conv2d_1.w_0.int8')
                          .get_tensor())

        self.assertEqual(w_8bit.dtype, np.int8)
        self.assertEqual(np.sum(w_8bit), np.sum(w_freeze))

    def test_freeze_program_cuda(self):
        self.freeze_program(True)


if __name__ == '__main__':
    unittest.main()
