import unittest
import paddle.v2.framework.core as core


class TestOpSupportGPU(unittest.TestCase):
    def test_case(self):
        self.assertEqual(core.is_compile_gpu(), core.op_support_gpu("sum"))


if __name__ == '__main__':
    unittest.main()
