#! python

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import os
import tempfile
import shutil
import sys
import importlib
from sampcd_processor import find_all
from sampcd_processor import check_indent
from sampcd_processor import get_api_md5
from sampcd_processor import get_incrementapi
from sampcd_processor import get_wlist
from sampcd_processor import sampcd_extract_to_file
from sampcd_processor import execute_samplecode

SAMPLECODE_TEMP_DIR = 'samplecode_temp'


class Test_find_all(unittest.TestCase):
    def test_find_none(self):
        self.assertEqual(0, len(find_all('hello', 'world')))

    def test_find_one(self):
        self.assertListEqual([0], find_all('hello', 'hello'))

    def test_find_two(self):
        self.assertListEqual([1, 15],
                             find_all(' hello, world; hello paddle!', 'hello'))


class Test_check_indent(unittest.TestCase):
    def test_no_indent(self):
        self.assertEqual(0, check_indent('hello paddle'))

    def test_indent_4_spaces(self):
        self.assertEqual(4, check_indent('    hello paddle'))

    def test_indent_1_tab(self):
        self.assertEqual(4, check_indent("\thello paddle"))


class Test_execute_samplecode(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(SAMPLECODE_TEMP_DIR):
            os.mkdir(SAMPLECODE_TEMP_DIR)
        self.successSampleCodeFile = os.path.join(SAMPLECODE_TEMP_DIR,
                                                  'samplecode_success.py')
        with open(self.successSampleCodeFile, 'w') as f:
            f.write('print(1+1)')
        self.failedSampleCodeFile = os.path.join(SAMPLECODE_TEMP_DIR,
                                                 'samplecode_failed.py')
        with open(self.failedSampleCodeFile, 'w') as f:
            f.write('print(1/0)')

    def tearDown(self):
        os.remove(self.successSampleCodeFile)
        os.remove(self.failedSampleCodeFile)

    def test_run_success(self):
        result, tfname, msg = execute_samplecode(self.successSampleCodeFile)
        self.assertTrue(result)
        self.assertEqual(self.successSampleCodeFile, tfname)
        self.assertIsNotNone(msg)
        self.assertLess(msg.find('skipped'), 0)

    def test_run_failed(self):
        result, tfname, msg = execute_samplecode(self.failedSampleCodeFile)
        self.assertFalse(result)
        self.assertEqual(self.failedSampleCodeFile, tfname)
        self.assertIsNotNone(msg)
        self.assertLess(msg.find('skipped'), 0)

    def test_testcases_skipped(self):
        ...
        tfname = os.path.join(SAMPLECODE_TEMP_DIR, 'samplecode_skipped.py')
        with open(tfname, 'w') as f:
            f.write("# required: distributed\nprint(1/0)")
        result, _, msg = execute_samplecode(tfname)
        self.assertTrue(result)
        self.assertGreaterEqual(msg.find('skipped'), 0)
        os.remove(tfname)


class Test_sampcd_extract_to_file(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(SAMPLECODE_TEMP_DIR):
            os.mkdir(SAMPLECODE_TEMP_DIR)

    def tearDown(self):
        shutil.rmtree(SAMPLECODE_TEMP_DIR)

    def test_1_samplecode(self):
        comments = """
        Examples:
            .. code-block:: python

                print(1+1)
        """
        funcname = 'one_plus_one'
        sample_code_filenames = sampcd_extract_to_file(comments, funcname)
        self.assertCountEqual(
            [os.path.join(SAMPLECODE_TEMP_DIR, funcname + '_example.py')],
            sample_code_filenames)

    def test_no_samplecode(self):
        comments = """
        placeholder
        """
        funcname = 'one_plus_one'
        sample_code_filenames = sampcd_extract_to_file(comments, funcname)
        self.assertCountEqual([], sample_code_filenames)

    def test_2_samplecodes(self):
        comments = """
        placeholder
        Examples:
            .. code-block:: python

                print(1/0)

            .. code-block:: python

                print(1+1)
        """
        funcname = 'one_plus_one'
        sample_code_filenames = sampcd_extract_to_file(comments, funcname)
        self.assertCountEqual([
            os.path.join(SAMPLECODE_TEMP_DIR, funcname + '_example_1.py'),
            os.path.join(SAMPLECODE_TEMP_DIR, funcname + '_example_2.py')
        ], sample_code_filenames)


class Test_get_api_md5(unittest.TestCase):
    def setUp(self):
        self.api_pr_spec_filename = os.path.abspath(
            os.path.join(os.getcwd(), "..", 'paddle/fluid/API_PR.spec'))
        with open(self.api_pr_spec_filename, 'w') as f:
            f.write("\n".join([
                """paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))""",
                """paddle.two_plus_two (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55two'))""",
                """paddle.three_plus_three (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6cthree'))""",
                """paddle.four_plus_four (paddle.four_plus_four, ('document', 'ff0f188c95030158cc6398d2a6c5four'))""",
            ]))

    def tearDown(self):
        os.remove(self.api_pr_spec_filename)
        pass

    def test_get_api_md5(self):
        res = get_api_md5('paddle/fluid/API_PR.spec')
        self.assertEqual("ff0f188c95030158cc6398d2a6c55one",
                         res['paddle.one_plus_one'])
        self.assertEqual("ff0f188c95030158cc6398d2a6c55two",
                         res['paddle.two_plus_two'])
        self.assertEqual("ff0f188c95030158cc6398d2a6cthree",
                         res['paddle.three_plus_three'])
        self.assertEqual("ff0f188c95030158cc6398d2a6c5four",
                         res['paddle.four_plus_four'])


class Test_get_incrementapi(unittest.TestCase):
    def setUp(self):
        self.api_pr_spec_filename = os.path.abspath(
            os.path.join(os.getcwd(), "..", 'paddle/fluid/API_PR.spec'))
        with open(self.api_pr_spec_filename, 'w') as f:
            f.write("\n".join([
                """paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))""",
                """paddle.two_plus_two (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55two'))""",
                """paddle.three_plus_three (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6cthree'))""",
                """paddle.four_plus_four (paddle.four_plus_four, ('document', 'ff0f188c95030158cc6398d2a6c5four'))""",
            ]))
        self.api_dev_spec_filename = os.path.abspath(
            os.path.join(os.getcwd(), "..", 'paddle/fluid/API_DEV.spec'))
        with open(self.api_dev_spec_filename, 'w') as f:
            f.write("\n".join([
                """paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))""",
            ]))
        self.api_diff_spec_filename = os.path.abspath(
            os.path.join(os.getcwd(), "dev_pr_diff_api.spec"))

    def tearDown(self):
        os.remove(self.api_pr_spec_filename)
        os.remove(self.api_dev_spec_filename)
        os.remove(self.api_diff_spec_filename)

    def test_it(self):
        get_incrementapi()
        with open(self.api_diff_spec_filename, 'r') as f:
            lines = f.readlines()
            self.assertCountEqual([
                "paddle.two_plus_two\n", "paddle.three_plus_three\n",
                "paddle.four_plus_four\n"
            ], lines)


class Test_get_wlist(unittest.TestCase):
    def setUp(self):
        self.tmpDir = tempfile.mkdtemp()
        self.wlist_filename = os.path.join(self.tmpDir, 'wlist.json')
        with open(self.wlist_filename, 'w') as f:
            f.write(r'''
{
    "wlist_dir":[
        {
            "name":"../python/paddle/fluid/contrib",
            "annotation":""
        },
        {
            "name":"../python/paddle/verison.py",
            "annotation":""
        }
    ],
    "wlist_api":[
        {
            "name":"xxxxx",
            "annotation":"not a real api, just for example"
        }
    ],
    "wlist_temp_api":[
        "to_tensor",
        "save_persistables@dygraph/checkpoint.py"
    ],
    "gpu_not_white":[
        "deformable_conv"
    ]
}
''')

    def tearDown(self):
        os.remove(self.wlist_filename)
        shutil.rmtree(self.tmpDir)

    def test_get_wlist(self):
        wlist, wlist_file, gpu_not_white = get_wlist(self.wlist_filename)
        self.assertCountEqual(
            ["xxxxx", "to_tensor",
             "save_persistables@dygraph/checkpoint.py"], wlist)
        self.assertCountEqual([
            "../python/paddle/fluid/contrib",
            "../python/paddle/verison.py",
        ], wlist_file)
        self.assertCountEqual(["deformable_conv"], gpu_not_white)


# https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/layers/ops.py
# why? unabled to use the ast module. emmmmm

if __name__ == '__main__':
    unittest.main()
