import paddle.v2.framework.core
import unittest


class TestScope(unittest.TestCase):
    def test_create_destroy(self):
        paddle_c = paddle.v2.framework.core
        scope = paddle_c.Scope(None)
        self.assertIsNotNone(scope)
        scope_with_parent = paddle_c.Scope(scope)
        self.assertIsNotNone(scope_with_parent)

    def test_none_variable(self):
        paddle_c = paddle.v2.framework.core
        scope = paddle_c.Scope(None)
        self.assertIsNone(scope.get_var("test"))

    def test_create_var_get_var(self):
        paddle_c = paddle.v2.framework.core
        scope = paddle_c.Scope(None)
        var_a = scope.create_var("var_a")
        self.assertIsNotNone(var_a)
        self.assertIsNotNone(scope.get_var('var_a'))
        scope2 = paddle_c.Scope(scope)
        self.assertIsNotNone(scope2.get_var('var_a'))

    def test_var_get_int(self):
        paddle_c = paddle.v2.framework.core
        scope = paddle_c.Scope(None)
        var = scope.create_var("test_int")
        var.set_int(10)
        self.assertTrue(var.is_int())
        self.assertEqual(10, var.get_int())


if __name__ == '__main__':
    unittest.main()
