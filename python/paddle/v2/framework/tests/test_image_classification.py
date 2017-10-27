import paddle.v2.framework.layers as layers
import paddle.v2.framework.nets as nets
from paddle.v2.framework.framework import Program
import paddle.v2.framework.core as core
import unittest


class TestBook(unittest.TestCase):
    def test_batch_norm_layer(self):
        program = Program()
        images = layers.data(
            name='pixel',
            shape=[3, 48, 48],
            data_type='float32',
            program=program)
        layers.batch_norm(input=images, program=program)

        print str(program)


if __name__ == '__main__':
    unittest.main()
