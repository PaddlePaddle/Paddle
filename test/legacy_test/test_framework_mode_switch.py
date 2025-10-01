# test/legacy_test/test_framework_mode_switch.py
import unittest
import paddle
from paddle.base.framework import in_dygraph_mode
from paddle.base.framework import on_static_mode, on_dygraph_mode


class TestModeSwitchCtx(unittest.TestCase):
    def setUp(self):
        self._was_dygraph = in_dygraph_mode()

    def tearDown(self):
        if self._was_dygraph:
            paddle.disable_static()
        else:
            paddle.enable_static()

    def test_on_static_mode_from_dygraph(self):
        paddle.disable_static()
        self.assertTrue(in_dygraph_mode())
        with on_static_mode():
            self.assertFalse(in_dygraph_mode())
        self.assertTrue(in_dygraph_mode())

    def test_on_static_mode_from_static(self):
        paddle.enable_static()
        self.assertFalse(in_dygraph_mode())
        with on_static_mode():
            self.assertFalse(in_dygraph_mode())
        self.assertFalse(in_dygraph_mode())

    def test_on_static_mode_force_static(self):
        paddle.disable_static()
        with on_static_mode(force_static=True):
            self.assertFalse(in_dygraph_mode())
        self.assertFalse(in_dygraph_mode())

    def test_on_dygraph_mode_from_static(self):
        paddle.enable_static()
        self.assertFalse(in_dygraph_mode())
        with on_dygraph_mode():
            self.assertTrue(in_dygraph_mode())
        self.assertFalse(in_dygraph_mode())

    def test_on_dygraph_mode_from_dygraph(self):
        paddle.disable_static()
        self.assertTrue(in_dygraph_mode())
        with on_dygraph_mode():
            self.assertTrue(in_dygraph_mode())
        self.assertTrue(in_dygraph_mode())

    def test_on_dygraph_mode_force_dygraph(self):
        paddle.enable_static()
        with on_dygraph_mode(force_dygraph=True):
            self.assertTrue(in_dygraph_mode())
        self.assertTrue(in_dygraph_mode())


if __name__ == "__main__":
    unittest.main()
