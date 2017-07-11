import paddle.v2.framework.c
import unittest


class TestScope(unittest.TestCase):
    def test_create_destroy(self):
        paddle_c = paddle.v2.framework.c
        scope = paddle_c.Scope()
        self.assertIsNotNone(scope)
        scope_with_parent = paddle_c.Scope(scope)
        self.assertIsNotNone(scope_with_parent)

    def test_none_variable(self):
        paddle_c = paddle.v2.framework.c
        scope = paddle_c.Scope()
        self.assertIsNone(scope.get_var("test"))

    def test_create_var_get_var(self):
        paddle_c = paddle.v2.framework.c
        scope = paddle_c.Scope()
        var_a = scope.create_var("var_a")
        self.assertIsNotNone(var_a)
        self.assertIsNotNone(scope.get_var('var_a'))
        scope2 = paddle_c.Scope(scope)
        self.assertIsNotNone(scope2.get_var('var_a'))


if __name__ == '__main__':
    unittest.main()
