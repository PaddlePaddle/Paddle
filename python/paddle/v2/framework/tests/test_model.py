from paddle.v2.framework.model import *
import unittest


class TestModel(unittest.TestCase):
    def test_simple_fc(self):
        img = data_layer(name="img", dims=784)
        hidden1 = fc_layer(input=img, size=200, act="sigmoid")
        hidden2 = fc_layer(input=hidden1, size=200, act="sigmoid")
        fc_layer(input=hidden2, size=10, act='softmax')
        init_params()
        forward({"img": numpy.random.random((100, 784)).astype('float32')})


if __name__ == '__main__':
    unittest.main()
