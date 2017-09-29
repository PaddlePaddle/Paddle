import paddle.v2.framework.core as core
import unittest


class TestException(unittest.TestCase):
    def test_exception(self):
        ex = None
        try:
            core.__unittest_throw_exception__()
        except core.EnforceNotMet as ex:
            self.assertIn("test exception", ex.message)

        self.assertIsNotNone(ex)


if __name__ == "__main__":
    unittest.main()
