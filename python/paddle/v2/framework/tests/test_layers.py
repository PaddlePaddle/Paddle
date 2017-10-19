import paddle.v2.framework.layers as layers
import paddle.v2.framework.nets as nets
from paddle.v2.framework.framework import Program, g_program
import paddle.v2.framework.core as core
import unittest


class TestBook(unittest.TestCase):
    def test_fit_a_line(self):
        program = Program()
        x = layers.data(
            name='x', shape=[13], data_type='float32', program=program)
        y_predict = layers.fc(input=x, size=1, act=None, program=program)

        y = layers.data(
            name='y', shape=[1], data_type='float32', program=program)
        cost = layers.square_error_cost(
            input=y_predict, label=y, program=program)

        avg_cost = layers.mean(x=cost, program=program)
        self.assertIsNotNone(avg_cost)
        program.append_backward(avg_cost)
        # print str(program)

    def test_recognize_digits_mlp(self):
        program = Program()

        # Change g_program, so the rest layers use `g_program`
        images = layers.data(
            name='pixel', shape=[784], data_type='float32', program=program)
        label = layers.data(
            name='label', shape=[1], data_type='int32', program=program)
        hidden1 = layers.fc(input=images, size=128, act='relu', program=program)
        hidden2 = layers.fc(input=hidden1, size=64, act='relu', program=program)
        predict = layers.fc(input=hidden2,
                            size=10,
                            act='softmax',
                            program=program)
        cost = layers.cross_entropy(input=predict, label=label, program=program)
        avg_cost = layers.mean(x=cost, program=program)
        self.assertIsNotNone(avg_cost)
        # print str(program)

    def test_simple_conv2d(self):
        program = Program()
        images = layers.data(
            name='pixel', shape=[3, 48, 48], data_type='int32', program=program)
        layers.conv2d(
            input=images, num_filters=3, filter_size=[4, 4], program=program)

        # print str(program)

    def test_recognize_digits_conv(self):
        program = Program()

        images = layers.data(
            name='pixel',
            shape=[1, 28, 28],
            data_type='float32',
            program=program)
        label = layers.data(
            name='label', shape=[1], data_type='int32', program=program)
        conv_pool_1 = nets.simple_img_conv_pool(
            input=images,
            filter_size=5,
            num_filters=2,
            pool_size=2,
            pool_stride=2,
            act="relu",
            program=program)
        conv_pool_2 = nets.simple_img_conv_pool(
            input=conv_pool_1,
            filter_size=5,
            num_filters=4,
            pool_size=2,
            pool_stride=2,
            act="relu",
            program=program)

        predict = layers.fc(input=conv_pool_2,
                            size=10,
                            act="softmax",
                            program=program)
        cost = layers.cross_entropy(input=predict, label=label, program=program)
        avg_cost = layers.mean(x=cost, program=program)

        program.append_backward(avg_cost)

        print str(program)


if __name__ == '__main__':
    unittest.main()
