import unittest
import paddle.v2.framework.core as core
import paddle.v2.framework.layers as layers


class TestArrayReadWrite(unittest.TestCase):
    def test_read_write(self):
        x = [
            layers.data(
                name='x0', shape=[100]), layers.data(
                    name='x1', shape=[100]), layers.data(
                        name='x2', shape=[100])
        ]

        i = layers.ones(shape=[1], dtype='int64')
        arr = layers.array_write(x=x[0], i=i)
        layers.increment(x=i)
        arr = layers.array_write(x=x[1], i=i, array=arr)
        layers.increment(x=i)
        arr = layers.array_write(x=x[2], i=i, array=arr)

        print arr.block.program


if __name__ == '__main__':
    unittest.main()
