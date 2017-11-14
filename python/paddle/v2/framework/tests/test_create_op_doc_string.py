import unittest
import paddle.v2.framework.layers as layers


class TestDocString(unittest.TestCase):
    def test_layer_doc_string(self):
        print layers.dropout.__doc__


if __name__ == '__main__':
    unittest.main()
