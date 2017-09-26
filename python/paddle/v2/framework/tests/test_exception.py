import paddle.v2.framework.core as core
import unittest


class TestException(unittest.TestCase):
    def test_exception(self):
        self.assertRaises(core.EnforceNotMet,
                          lambda: core.__unittest_throw_exception__())


if __name__ == "__main__":
    unittest.main()
