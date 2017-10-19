from paddle.v2.framework.default_scope_funcs import *
import unittest


class TestDefaultScopeFuncs(unittest.TestCase):
    def test_cur_scope(self):
        self.assertIsNotNone(get_cur_scope())

    def test_none_variable(self):
        self.assertIsNone(find_var("test"))

    def test_create_var_get_var(self):
        var_a = var("var_a")
        self.assertIsNotNone(var_a)
        self.assertIsNotNone(get_cur_scope().find_var('var_a'))
        enter_local_scope()
        self.assertIsNotNone(get_cur_scope().find_var('var_a'))
        leave_local_scope()

    def test_var_get_int(self):
        def __new_scope__():
            i = var("var_i")
            self.assertFalse(i.is_int())
            i.set_int(10)
            self.assertTrue(i.is_int())
            self.assertEqual(10, i.get_int())

        for _ in xrange(10):
            scoped_function(__new_scope__)


if __name__ == '__main__':
    unittest.main()
