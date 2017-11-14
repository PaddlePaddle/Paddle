import unittest
from paddle.v2.framework.framework import Program


class TestDebugStringFramework(unittest.TestCase):
    def test_debug_str(self):
        p = Program()
        p.current_block().create_var(name='t', shape=[0, 1])
        self.assertRaises(ValueError, callableObj=p.__str__)


if __name__ == '__main__':
    unittest.main()
