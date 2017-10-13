from paddle.v2.framework.layers import fc_layer, data_layer, cross_entropy
import unittest


class TestFCLayer(unittest.TestCase):
    def test_mnist(self):
        img = data_layer(name="img", shape=[28 * 28], data_type='float32')
        hidden = fc_layer(input=img, size=200, act="tanh")
        hidden = fc_layer(input=hidden, size=200, act="tanh")
        inference = fc_layer(input=hidden, size=10, act="softmax")
        cost = cross_entropy(
            input=inference,
            label=data_layer(
                name='label', shape=[1], data_type='int32'))
        self.assertIsNotNone(cost)
        print str(cost.block.program)


if __name__ == '__main__':
    unittest.main()
