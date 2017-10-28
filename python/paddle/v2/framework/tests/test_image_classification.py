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
        images = layers.data(
            name='pixel',
            shape=[3, 48, 48],
            data_type='float32',
            program=program)
        layers.batch_norm(input=images, program=program)

        #print str(program)

    def test_dropout_layer(self):
        program = Program()
        images = layers.data(
            name='pixel',
            shape=[3, 48, 48],
            data_type='float32',
            program=program)
        layers.dropout(x=images, dropout_prob=0.5, program=program)

        #print str(program)

    def test_img_conv_group(self):
        classdim = 10

        program = Program()
        images = layers.data(
            name='pixel',
            shape=[3, 48, 48],
            data_type='float32',
            program=program)
        label = layers.data(
            name='label',
            shape=[1],
            data_type='int32',
            program=program,
            init_program=init_program)
        conv1 = conv_block(images, 64, 2, [0.3, 0], program)
        conv2 = conv_block(conv1, 128, 2, [0.4, 0], program)
        conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0], program)
        conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0], program)
        conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0], program)

        drop = layers.dropout(x=conv5, dropout_prob=0.5, program=program)
        fc1 = layers.fc(input=drop, size=512, act=None, program=program)
        bn = layers.batch_norm(input=fc1, act='relu', program=program)
        drop2 = layers.dropout(x=bn, dropout_prob=0.5, program=program)
        fc2 = layers.fc(input=drop2, size=512, act=None, program=program)
        out = layers.fc(input=fc2,
                        size=classdim,
                        act='softmax',
                        program=program)

        print str(program)


class TestBook(unittest.TestCase):
    def test_chapter_3(self):
        pass


if __name__ == '__main__':
    unittest.main()
