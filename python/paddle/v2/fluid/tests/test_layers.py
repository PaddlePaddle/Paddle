from __future__ import print_function
import unittest

import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.nets as nets
from paddle.v2.fluid.framework import Program, program_guard
from paddle.v2.fluid.param_attr import ParamAttr


class TestBook(unittest.TestCase):
    def test_fit_a_line(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            x = layers.data(name='x', shape=[13], dtype='float32')
            y_predict = layers.fc(input=x, size=1, act=None)
            y = layers.data(name='y', shape=[1], dtype='float32')
            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(x=cost)
            self.assertIsNotNone(avg_cost)
            program.append_backward(avg_cost)

        print(str(program))

    def test_recognize_digits_mlp(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            # Change g_program, so the rest layers use `g_program`
            images = layers.data(name='pixel', shape=[784], dtype='float32')
            label = layers.data(name='label', shape=[1], dtype='int32')
            hidden1 = layers.fc(input=images, size=128, act='relu')
            hidden2 = layers.fc(input=hidden1, size=64, act='relu')
            predict = layers.fc(input=hidden2, size=10, act='softmax')
            cost = layers.cross_entropy(input=predict, label=label)
            avg_cost = layers.mean(x=cost)
            self.assertIsNotNone(avg_cost)

        print(str(program))

    def test_simple_conv2d(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            images = layers.data(name='pixel', shape=[3, 48, 48], dtype='int32')
            layers.conv2d(input=images, num_filters=3, filter_size=[4, 4])

        print(str(program))

    def test_conv2d_transpose(self):
        program = Program()
        with program_guard(program):
            img = layers.data(name='pixel', shape=[3, 2, 2], dtype='float32')
            layers.conv2d_transpose(input=img, num_filters=10, output_size=28)
        print(str(program))

    def test_recognize_digits_conv(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            images = layers.data(
                name='pixel', shape=[1, 28, 28], dtype='float32')
            label = layers.data(name='label', shape=[1], dtype='int32')
            conv_pool_1 = nets.simple_img_conv_pool(
                input=images,
                filter_size=5,
                num_filters=2,
                pool_size=2,
                pool_stride=2,
                act="relu")
            conv_pool_2 = nets.simple_img_conv_pool(
                input=conv_pool_1,
                filter_size=5,
                num_filters=4,
                pool_size=2,
                pool_stride=2,
                act="relu")

            predict = layers.fc(input=conv_pool_2, size=10, act="softmax")
            cost = layers.cross_entropy(input=predict, label=label)
            avg_cost = layers.mean(x=cost)

            program.append_backward(avg_cost)

        print(str(program))

    def test_word_embedding(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            dict_size = 10000
            embed_size = 32
            first_word = layers.data(name='firstw', shape=[1], dtype='int64')
            second_word = layers.data(name='secondw', shape=[1], dtype='int64')
            third_word = layers.data(name='thirdw', shape=[1], dtype='int64')
            forth_word = layers.data(name='forthw', shape=[1], dtype='int64')
            next_word = layers.data(name='nextw', shape=[1], dtype='int64')

            embed_first = layers.embedding(
                input=first_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')
            embed_second = layers.embedding(
                input=second_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')

            embed_third = layers.embedding(
                input=third_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')
            embed_forth = layers.embedding(
                input=forth_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')

            concat_embed = layers.concat(
                input=[embed_first, embed_second, embed_third, embed_forth],
                axis=1)

            hidden1 = layers.fc(input=concat_embed, size=256, act='sigmoid')
            predict_word = layers.fc(input=hidden1,
                                     size=dict_size,
                                     act='softmax')
            cost = layers.cross_entropy(input=predict_word, label=next_word)
            avg_cost = layers.mean(x=cost)
            self.assertIsNotNone(avg_cost)

        print(str(program))

    def test_linear_chain_crf(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            label_dict_len = 10
            images = layers.data(name='pixel', shape=[784], dtype='float32')
            label = layers.data(name='label', shape=[1], dtype='int32')
            hidden = layers.fc(input=images, size=128)
            crf = layers.linear_chain_crf(
                input=hidden, label=label, param_attr=ParamAttr(name="crfw"))
            crf_decode = layers.crf_decoding(
                input=hidden, param_attr=ParamAttr(name="crfw"))
            layers.chunk_eval(
                input=crf_decode,
                label=label,
                chunk_scheme="IOB",
                num_chunk_types=(label_dict_len - 1) / 2)
            self.assertNotEqual(crf, None)
            self.assertNotEqual(crf_decode, None)

        print(str(program))

    def test_sigmoid_cross_entropy(self):
        program = Program()
        with program_guard(program):
            dat = layers.data(name='data', shape=[10], dtype='float32')
            lbl = layers.data(name='label', shape=[10], dtype='float32')
            self.assertIsNotNone(
                layers.sigmoid_cross_entropy_with_logits(
                    x=dat, label=lbl))
        print(str(program))


if __name__ == '__main__':
    unittest.main()
