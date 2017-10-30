import unittest

import paddle.v2.framework.layers as layers
import paddle.v2.framework.nets as nets
from paddle.v2.framework.framework import Program


def conv_block(input,
               num_filter,
               groups,
               dropouts,
               program=None,
               init_program=None):
    return nets.img_conv_group(
        input=input,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[num_filter] * groups,
        conv_filter_size=3,
        conv_act='relu',
        conv_with_batchnorm=True,
        conv_batchnorm_drop_rate=dropouts,
        pool_type='max',
        program=program,
        init_program=init_program)


class TestLayer(unittest.TestCase):
    def test_batch_norm_layer(self):
        program = Program()
        init_program = Program()
        images = layers.data(
            name='pixel',
            shape=[3, 48, 48],
            data_type='float32',
            program=program)
        layers.batch_norm(
            input=images, program=program, init_program=init_program)

        #print str(program)

    def test_dropout_layer(self):
        program = Program()
        init_program = Program()
        images = layers.data(
            name='pixel',
            shape=[3, 48, 48],
            data_type='float32',
            program=program)
        layers.dropout(
            x=images,
            dropout_prob=0.5,
            program=program,
            init_program=init_program)

        #print str(program)

    def test_img_conv_group(self):
        program = Program()
        init_program = Program()

        images = layers.data(
            name='pixel',
            shape=[3, 48, 48],
            data_type='float32',
            program=program,
            init_program=init_program)
        conv1 = conv_block(images, 64, 2, [0.3, 0], program, init_program)
        conv2 = conv_block(conv1, 256, 3, [0.4, 0.4, 0], program, init_program)

        # print str(program)

    def test_elementwise_add_with_act(self):
        program = Program()
        init_program = Program()
        image1 = layers.data(
            name='pixel1',
            shape=[3, 48, 48],
            data_type='float32',
            program=program,
            init_program=init_program)
        image2 = layers.data(
            name='pixel2',
            shape=[3, 48, 48],
            data_type='float32',
            program=program,
            init_program=init_program)
        out = layers.elementwise_add(
            x=image1,
            y=image2,
            act='relu',
            program=program,
            init_program=init_program)
        # print(program)


if __name__ == '__main__':
    unittest.main()
