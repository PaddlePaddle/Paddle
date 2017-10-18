from paddle.v2.framework.layers import fc_layer, data_layer, cross_entropy, mean, square_error_cost
from paddle.v2.framework.framework import Program, g_program
import paddle.v2.framework.core as core
import unittest


class TestBook(unittest.TestCase):
    def test_fit_a_line(self):
        program = Program()
        x = data_layer(
            name='x', shape=[13], data_type='float32', program=program)
        y_predict = fc_layer(input=x, size=1, act=None, program=program)

        y = data_layer(
            name='y', shape=[1], data_type='float32', program=program)
        cost = square_error_cost(input=y_predict, label=y, program=program)

        avg_cost = mean(x=cost, program=program)
        self.assertIsNotNone(avg_cost)
        print str(program)

    def test_recognize_digits_mlp(self):
        program = Program()

        # Change g_program, so the rest layers use `g_program`
        images = data_layer(
            name='pixel', shape=[784], data_type='float32', program=program)
        label = data_layer(
            name='label', shape=[1], data_type='int32', program=program)
        hidden1 = fc_layer(input=images, size=128, act='relu', program=program)
        hidden2 = fc_layer(input=hidden1, size=64, act='relu', program=program)
        predict = fc_layer(
            input=hidden2, size=10, act='softmax', program=program)
        cost = cross_entropy(input=predict, label=label, program=program)
        avg_cost = mean(x=cost, program=program)
        self.assertIsNotNone(avg_cost)
        print str(program)


if __name__ == '__main__':
    unittest.main()
