import unittest
from paddle.v2.framework.framework import g_main_program
import paddle.v2.framework.core as core


class TestParameter(unittest.TestCase):
    def test_param(self):
        b = g_main_program.create_block()
        param = b.create_parameter(
            name='fc.w',
            shape=[784, 100],
            dtype='float32',
            initialize_attr={
                'type': 'uniform_random',
                'seed': 13,
                'min': -5.0,
                'max': 5.0
            })
        self.assertIsNotNone(param)
        self.assertEqual('fc.w', param.name)
        self.assertEqual((784, 100), param.shape)
        self.assertEqual(core.DataType.FP32, param.data_type)
        self.assertEqual(0, param.block.idx)


if __name__ == '__main__':
    unittest.main()
