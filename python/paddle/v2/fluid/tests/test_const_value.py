import unittest
import paddle.v2.fluid.framework as framework


class ConditionalBlock(unittest.TestCase):
    def test_const_value(self):
        self.assertEqual(framework.GRAD_VAR_SUFFIX, "@GRAD")
        self.assertEqual(framework.TEMP_VAR_NAME, "@TEMP@")
        self.assertEqual(framework.GRAD_VAR_SUFFIX, "@GRAD")
        self.assertEqual(framework.ZERO_VAR_SUFFIX, "@ZERO")


if __name__ == '__main__':
    unittest.main()
