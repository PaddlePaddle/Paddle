# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

import argparse
import itertools
import re
import sys
import unittest
from typing import List, Union
from unittest import suite

parser = argparse.ArgumentParser(description="Argparse for op test helper")
parser.add_argument(
    "--case",
    type=str,
    help="Which case you want to test, default -1 for all cases.",
    default=None,
)
args = parser.parse_args()


class TestCaseHelper:
    """
    Helper class for constructing test cases.
    """

    def __init__(self):
        self.custom_attrs_list = []

    def init_attrs(self):
        """
        Initialize attributes for op
        """
        raise Exception("Not implemented.")

    def _flatten_tuple(self, cur_tuple):
        """
        Expand the nested dict in tuple
        """
        new_dict = []
        for cur_dict in cur_tuple:
            for k, v in cur_dict.items():
                new_dict.append((k, v))
        return dict(new_dict)

    def _register_custom_attrs(self, custom_attrs):
        """
        register custom attribute
        """
        self.custom_attrs_list.append(custom_attrs)

    def _init_cases(self):
        """
        Generate all test cases
        """
        assert isinstance(self.inputs, list)
        assert isinstance(self.dtypes, list)
        assert isinstance(self.attrs, list)
        self.all_cases = []
        all_lists = [
            self.inputs,
            self.dtypes,
            self.attrs,
            *self.custom_attrs_list,
        ]
        filtered_lists = filter(lambda x: len(x) > 0, all_lists)
        for case in itertools.product(*filtered_lists):
            self.all_cases.append(self._flatten_tuple(case))

    def _make_all_classes(self):
        """
        Generate test classes
        """
        self.init_attrs()
        self._init_cases()
        self.all_classes = []
        if args.case is not None:
            for test_name in self.specify_test:
                no = int(re.search(r'\d+$', test_name).group(0))
                assert 0 <= no and no < len(self.all_cases)
                self.all_classes.append(
                    type(
                        f'{self.__class__.__name__}.{self.class_name}{no}',
                        (self.cls,),
                        {"case": self.all_cases[no]},
                    )
                )
        else:
            for i, case in enumerate(self.all_cases):
                self.all_classes.append(
                    type(
                        f'{self.__class__.__name__}.{self.class_name}{i}',
                        (self.cls,),
                        {"case": case},
                    )
                )

    def run(self):
        """
        Run all test classes
        """
        if args.case is not None:
            self.specify_test = []
            all_tests = args.case.split(',')
            for test in all_tests:
                test_info = test.split('.')
                assert len(test_info) == 2
                if self.__class__.__name__ == test_info[0]:
                    self.specify_test.append(test_info[1])
            if len(self.specify_test) == 0:
                return
        self._make_all_classes()
        test_suite = unittest.TestSuite()
        test_loader = unittest.TestLoader()
        for x in self.all_classes:
            test_suite.addTests(test_loader.loadTestsFromTestCase(x))
        runner = unittest.TextTestRunner()
        res = runner.run(test_suite)
        if not res.wasSuccessful():
            sys.exit(not res.wasSuccessful())


def run_test(test_class: Union[suite.TestSuite, List[suite.TestSuite]]):
    test_suite = unittest.TestSuite()
    test_loader = unittest.TestLoader()
    if isinstance(test_class, type):
        test_suite.addTests(test_loader.loadTestsFromTestCase(test_class))
    else:
        for cls in test_class:
            test_suite.addTests(test_loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner()
    res = runner.run(test_suite)
    if not res.wasSuccessful():
        sys.exit(not res.wasSuccessful())
