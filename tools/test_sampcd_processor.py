#! python
import unittest
from sampcd_processor import find_all
from sampcd_processor import check_indent

class Test_find_all(unittest.TestCase):
    # def test_srcstr_is_None(self):
    #    self.assertIsNone(find_all(None, 'hello world'))
    def test_find_none(self):
        self.assertEqual(0, len(find_all('hello', 'world')))
    def test_find_one(self):
        self.assertListEqual([0], find_all('hello', 'hello'))
    def test_find_two(self):
        self.assertListEqual([1, 15], find_all(' hello, world; hello paddle!', 'hello'))

class Test_check_indent(unittest.TestCase):
    def test_no_indent(self):
        self.assertEqual(0, check_indent('hello paddle'))
    def test_indent_4_spaces(self):
        self.assertEqual(4, check_indent('    hello paddle'))
    def test_indent_1_tab(self):
        self.assertEqual(4, check_indent("\thello paddle"))
    # def test_indent_mixed_spaces_and_tab(self):
    #     with self.assertRaises(Exception):
    #         check_indent("  \thello paddle")


if __name__ == '__main__':
    unittest.main()
