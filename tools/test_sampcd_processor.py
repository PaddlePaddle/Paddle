#! python
import unittest
import os
import tempfile
import shutil
import sys
import importlib
from sampcd_processor import find_all
from sampcd_processor import check_indent
from sampcd_processor import sampcd_extract_and_run
from sampcd_processor import single_defcom_extract
from sampcd_processor import srccoms_extract
from sampcd_processor import get_api_md5

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

class Test_sampcd_extract_and_run(unittest.TestCase):
    def setUp(self):
        if not os.path.exists('samplecode_temp/'):
            os.mkdir('samplecode_temp/')
    def test_run_a_defs_samplecode(self):
        comments = """
        Examples:
            .. code-block:: python
                print(1+1)
        """
        funcname = 'one_plus_one'
        self.assertTrue(sampcd_extract_and_run(comments, funcname))
    def test_run_a_def_no_code(self):
        comments = """
        placeholder
        """
        funcname = 'one_plus_one'
        self.assertFalse(sampcd_extract_and_run(comments, funcname))
    def test_run_a_def_raise_expection(self):
        comments = """
        placeholder
        Examples:
            .. code-block:: python
                print(1/0)
        """
        funcname = 'one_plus_one'
        self.assertFalse(sampcd_extract_and_run(comments, funcname))

class Test_single_defcom_extract(unittest.TestCase):
    def test_extract_from_func(self):
        defstr='''
import os
def foo():
            """
            foo is a function.
            """
            pass
def bar():
            pass
'''
        comm = single_defcom_extract(2, defstr.splitlines(True), is_class_begin=False)
        self.assertEqual("            foo is a function.\n", comm)
        pass
    def test_extract_from_func_with_no_docstring(self):
        defstr='''
import os
def bar():
            pass
'''
        comm = single_defcom_extract(2, defstr.splitlines(True), is_class_begin=False)
        self.assertEqual('', comm)
        pass
    def test_extract_from_class(self):
        defstr=r'''
import os
class Foo():
            """
            Foo is a class.
            second line.
            """
            pass
            def bar():
                pass
def foo():
            pass
'''
        comm = single_defcom_extract(2, defstr.splitlines(True), is_class_begin=True)
        rcomm = """            Foo is a class.
            second line.
"""
        self.assertEqual(rcomm, comm)
        pass
    def test_extract_from_class_with_no_docstring(self):
        defstr='''
import os
class Foo():
            pass
            def bar():
                pass
def foo():
            pass
'''
        comm = single_defcom_extract(0, defstr.splitlines(True), is_class_begin=True)
        self.assertEqual('', comm)

class Test_get_api_md5(unittest.TestCase):
    def setUp(self):
        self.api_pr_spec_filename = os.path.abspath(os.path.join(os.getcwd(), "..",'paddle/fluid/API_PR.spec'))
        with open(self.api_pr_spec_filename, 'w') as f:
            f.write("\n".join([
                """one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'one_plus_one'))""",
                """two_plus_two (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'two_plus_two'))""",
                """three_plus_three (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'three_plus_three'))""",
                """four_plus_four (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'four_plus_four'))""",
                ]))
    def tearDown(self):
        os.remove(self.api_pr_spec_filename)
        pass
    def test_get_api_md5(self):
        res = get_api_md5('paddle/fluid/API_PR.spec')
        self.assertEqual("'one_plus_one'", res['one_plus_one'])
        self.assertEqual("'two_plus_two'", res['two_plus_two'])
        self.assertEqual("'three_plus_three'", res['three_plus_three'])
        self.assertEqual("'four_plus_four'", res['four_plus_four'])

class Test_srccoms_extract(unittest.TestCase):
    def setUp(self):
        self.tmpDir = tempfile.mkdtemp()
        sys.path.append(self.tmpDir)
        self.api_pr_spec_filename = os.path.abspath(os.path.join(os.getcwd(), "..",'paddle/fluid/API_PR.spec'))
        with open(self.api_pr_spec_filename, 'w') as f:
            f.write("\n".join([
                """one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', "one_plus_one"))""",
                """two_plus_two (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', "two_plus_two"))""",
                """three_plus_three (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', "three_plus_three"))""",
                """four_plus_four (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', "four_plus_four"))""",
                ]))
    def tearDown(self):
        sys.path.remove(self.tmpDir)
        shutil.rmtree(self.tmpDir)
        os.remove(self.api_pr_spec_filename)
    def test_from_ops_py(self):
        filecont = '''
__all__ = []
__all__ += ['one_plus_one']

def add_sample_code(obj, docstr):
    pass
def one_plus_one():
            return 1+1

one_plus_one.__doc__ = """
            placeholder

            Examples:
            .. code-block:: python
                print(1+1)
"""

__all__ += ['two_plus_two']
def two_plus_two():
            return 2+2
add_sample_code(globals()["two_plus_two"], """
            Examples:
            .. code-block:: python
                print(2+2)
""")
'''
        pyfilename = os.path.join(self.tmpDir, 'ops.py')
        with open(pyfilename, 'w') as pyfile:
            pyfile.write(filecont)
        self.assertTrue(os.path.exists(pyfilename))
        utsp = importlib.import_module('ops')
        methods = ['one_plus_one','two_plus_two']
        with open(pyfilename, 'r') as pyfile:
            self.assertTrue(srccoms_extract(pyfile, [], methods))
        self.assertTrue(os.path.exists("samplecode_temp/" "one_plus_one_example.py")) 
        self.assertTrue(os.path.exists("samplecode_temp/" "two_plus_two_example.py"))
    def test_from_not_ops_py(self):
        filecont = '''
__all__ = [
        'one_plus_one'
]

def one_plus_one():
            """
            placeholder

            Examples:
            .. code-block:: python
                print(1+1)
            """
            return 1+1

'''
        pyfilename = os.path.join(self.tmpDir, 'opo.py')
        with open(pyfilename, 'w') as pyfile:
            pyfile.write(filecont)
        utsp = importlib.import_module('opo')
        methods = ['one_plus_one']
        with open(pyfilename, 'r') as pyfile:
            self.assertTrue(srccoms_extract(pyfile, [], methods))
        self.assertTrue(os.path.exists("samplecode_temp/" "one_plus_one_example.py"))
    def test_with_empty_wlist(self):
        """
        see test_from_ops_py
        """
        pass
    def test_with_wlist(self):
        filecont = '''
__all__ = [
        'four_plus_four',
        'three_plus_three'
        ]

def four_plus_four():
            """
            placeholder

            Examples:
            .. code-block:: python
                print(4+4)
            """
            return 4+4
def three_plus_three():
            """
            placeholder

            Examples:
            .. code-block:: python
                print(3+3)
            """
            return 3+3

'''
        pyfilename = os.path.join(self.tmpDir, 'three_and_four.py')
        with open(pyfilename, 'w') as pyfile:
            pyfile.write(filecont)
        utsp = importlib.import_module('three_and_four')
        methods = [
        'four_plus_four',
        'three_plus_three'
        ]
        with open(pyfilename, 'r') as pyfile:
            self.assertTrue(srccoms_extract(pyfile, ['three_plus_three'], methods))
        self.assertTrue(os.path.exists("samplecode_temp/four_plus_four_example.py"))
        self.assertFalse(os.path.exists("samplecode_temp/three_plus_three_example.py"))

# https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/layers/ops.py
# why? unabled to use the ast module. emmmmm

if __name__ == '__main__':
    unittest.main()
