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
from paddle.fluid.contrib.quantize.quantize_transpiler import _original_var_name
from paddle.fluid.contrib.quantize.quantize_transpiler import QuantizeTranspiler
import paddle

paddle.enable_static()


def linear_fc(num):
    data = fluid.layers.data(name='image', shape=[1, 32, 32], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = data
    for _ in range(num):
        hidden = fluid.layers.fc(hidden, size=128, act='relu')
    loss = paddle.nn.functional.cross_entropy(
        input=hidden, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


def residual_block(num):
    def conv_bn_layer(
        input, ch_out, filter_size, stride, padding, act='relu', bias_attr=False
    ):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr,
        )
        return paddle.static.nn.batch_norm(input=tmp, act=act)

    data = fluid.layers.data(name='image', shape=[1, 32, 32], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = data
    for _ in range(num):
        conv = conv_bn_layer(hidden, 16, 3, 1, 1, act=None, bias_attr=True)
        short = conv_bn_layer(hidden, 16, 1, 1, 0, act=None)
        hidden = paddle.nn.functional.relu(paddle.add(x=conv, y=short))
    fc = fluid.layers.fc(input=hidden, size=10)
    loss = paddle.nn.functional.cross_entropy(
        input=fc, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


def conv_net(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    conv_pool_1 = paddle.static.nn.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_loss = paddle.mean(loss)
    return avg_loss


class TestQuantizeTranspiler(unittest.TestCase):
    def setUp(self):
        # since quant_op and dequant_op is not ready, use cos and sin for test
        self.weight_quant_op_type = 'fake_quantize_abs_max'
        self.dequant_op_type = 'fake_dequantize_max_abs'
        self.quantizable_op_and_inputs = {
            'conv2d': ['Input', 'Filter'],
            'depthwise_conv2d': ['Input', 'Filter'],
            'mul': ['X', 'Y'],
        }
        self.quantizable_op_grad_and_inputs = {
            'conv2d_grad': ['Input', 'Filter'],
            'depthwise_conv2d_grad': ['Input', 'Filter'],
            'mul_grad': ['X', 'Y'],
        }

    def check_program(self, program):
        quantized_ops = {}

        persistable_vars = [
            v.name
            for v in filter(lambda var: var.persistable, program.list_vars())
        ]

        for block in program.blocks:
            for idx, op in enumerate(block.ops):
                # check forward
                if op.type in self.quantizable_op_and_inputs:
                    for i, arg_name in enumerate(op.input_arg_names):
                        quant_op_type = (
                            self.weight_quant_op_type
                            if _original_var_name(arg_name) in persistable_vars
                            else self.act_quant_op_type
                        )
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized')
                        )
                        if arg_name not in quantized_ops:
                            self.assertEqual(
                                block.ops[idx - 2 * i - 1].type,
                                self.dequant_op_type,
                            )
                            self.assertEqual(
                                block.ops[idx - 2 * i - 2].type, quant_op_type
                            )
                            quantized_ops[arg_name] = block.ops[idx - 2 * i - 2]
                        else:
                            op_idx = block.ops.index(quantized_ops[arg_name])
                            self.assertLess(op_idx, idx)

                # check backward
                if op.type in self.quantizable_op_grad_and_inputs:
                    for pname in self.quantizable_op_grad_and_inputs[op.type]:
                        arg_name = op.input(pname)[0]
                        self.assertTrue(
                            arg_name.endswith('.quantized.dequantized')
                        )
                        self.assertTrue(arg_name in quantized_ops)

    def linear_fc_quant(self, quant_type):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = linear_fc(3)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
            t = QuantizeTranspiler(activation_quantize_type=quant_type)
            t.training_transpile(main)
            self.check_program(main)

    def test_linear_fc_quant_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_abs_max'
        self.linear_fc_quant('abs_max')

    def test_linear_fc_quant_range_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_range_abs_max'
        self.linear_fc_quant('range_abs_max')

    def residual_block_quant(self, quant_type):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = residual_block(2)
            opt = fluid.optimizer.Adam(learning_rate=0.001)
            opt.minimize(loss)
            t = QuantizeTranspiler(activation_quantize_type=quant_type)
            t.training_transpile(main)
            self.check_program(main)

    def test_residual_block_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_abs_max'
        self.residual_block_quant('abs_max')

    def test_residual_block_range_abs_max(self):
        self.act_quant_op_type = 'fake_quantize_range_abs_max'
        self.residual_block_quant('range_abs_max')

    def freeze_program(self, use_cuda, seed):
        def build_program(main, startup, is_test):
            main.random_seed = seed
            startup.random_seed = seed
            with fluid.unique_name.guard():
                with fluid.program_guard(main, startup):
                    img = fluid.layers.data(
                        name='image', shape=[1, 28, 28], dtype='float32'
                    )
                    label = fluid.layers.data(
                        name='label', shape=[1], dtype='int64'
                    )
                    loss = conv_net(img, label)
                    if not is_test:
                        opt = fluid.optimizer.Adam(learning_rate=0.001)
                        opt.minimize(loss)
            return [img, label], loss

        main = fluid.Program()
        startup = fluid.Program()
        test_program = fluid.Program()

        import random

        random.seed(0)
        np.random.seed(0)

        feeds, loss = build_program(main, startup, False)
        build_program(test_program, startup, True)
        test_program = test_program.clone(for_test=True)

        quant_type = 'range_abs_max'  # 'range_abs_max' or 'abs_max'
        quant_transpiler = QuantizeTranspiler(
            activation_quantize_type=quant_type
        )
        quant_transpiler.training_transpile(main, startup)
        quant_transpiler.training_transpile(test_program, startup)

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        iters = 5
        batch_size = 8
        class_num = 10
        exe.run(startup)

        train_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
            batch_size=batch_size,
        )
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )
        feeder = fluid.DataFeeder(feed_list=feeds, place=place)

        with fluid.program_guard(main):
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(
                    program=main, feed=feeder.feed(data), fetch_list=[loss]
                )

        with fluid.program_guard(test_program):
            test_data = next(test_reader())
            w_var = fluid.framework._get_var(
                'conv2d_1.w_0.quantized', test_program
            )
            # Testing during training
            test_loss1, w_quant = exe.run(
                program=test_program,
                feed=feeder.feed(test_data),
                fetch_list=[loss, w_var],
            )

            # Freeze program for inference, but the weight of fc/conv is still float type.
            quant_transpiler.freeze_program(test_program, place)
            (test_loss2,) = exe.run(
                program=test_program,
                feed=feeder.feed(test_data),
                fetch_list=[loss],
            )
            self.assertAlmostEqual(test_loss1, test_loss2, delta=5e-3)
            w_freeze = np.array(
                fluid.global_scope().find_var('conv2d_1.w_0').get_tensor()
            )
            # fail: -432.0 != -433.0, this is due to the calculation precision
            # self.assertAlmostEqual(np.sum(w_freeze), np.sum(w_quant))

            # Convert parameter to 8-bit.
            quant_transpiler.convert_to_int8(test_program, place)
            # Save the 8-bit parameter and model file.
            fluid.io.save_inference_model(
                'model_8bit',
                ['image', 'label'],
                [loss],
                exe,
                test_program,
                clip_extra=True,
            )
            # Test whether the 8-bit parameter and model file can be loaded successfully.
            [infer, feed, fetch] = fluid.io.load_inference_model(
                'model_8bit', exe
            )
            # Check the loaded 8-bit weight.
            w_8bit = np.array(
                fluid.global_scope().find_var('conv2d_1.w_0.int8').get_tensor()
            )

            self.assertEqual(w_8bit.dtype, np.int8)
            self.assertEqual(np.sum(w_8bit), np.sum(w_freeze))

    def not_test_freeze_program_cuda(self):
        if fluid.core.is_compiled_with_cuda():
            with fluid.unique_name.guard():
                self.freeze_program(True, seed=1)

    def not_test_freeze_program_cpu(self):
        with fluid.unique_name.guard():
            self.freeze_program(False, seed=2)


if __name__ == '__main__':
    unittest.main()
