from __future__ import print_function
import unittest
import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.framework import Program, program_guard, KERNEL_HINT_KEY


class TestKernelHint(unittest.TestCase):
    def test_kernel_hint(self):
        kernel_hint = "cudnn"
        program = Program()
        with program_guard(program, startup_program=Program()):
            images = layers.data(name='pixel', shape=[3, 48, 48], dtype='int32')
            layers.conv2d(
                input=images,
                num_filters=3,
                filter_size=[4, 4],
                kernel_hint=kernel_hint)
        conv_op = filter(lambda op: op.type == "conv2d_cudnn",
                         program.block(0).ops)
        self.assertEqual(len(conv_op), 1)
        self.assertEqual(conv_op[0].attr(KERNEL_HINT_KEY), kernel_hint)


if __name__ == '__main__':
    unittest.main()
