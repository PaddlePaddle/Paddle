#!/bin/env python
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
""" Build file and unit test mapping. """

import os
import re
import json
import time
import os.path
import subprocess
from github import Github

PADDLE_ROOT = '{}/'.format(os.getenv('PADDLE_ROOT', '/paddle/')).replace('//',
                                                                         '/')


class UTMapper(object):
    """ Unit test mapper. """

    def __init__(self):
        self.github = Github(os.getenv('GITHUB_API_TOKEN'), timeout=60)
        self.repo = None
        self.ut_list = []
        self.src_ut_dict = {}

    def load_ctest_ut_list(self):
        """ Load ctest unit test list. """
        ps = subprocess.Popen(
            "ctest -N | awk -F ':' '{print $2}' | sed '/^$/d' | sed '$d' | sed 's/ //g'",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd='{}build'.format(PADDLE_ROOT))
        uts = ps.communicate()[0]
        self.ut_list = uts.rstrip('\n').encode().split('\n')

    def load_cpp_and_cuda_ut(self):
        """ Load C++ and CUDA unit test list. """
        data = None
        cpp_cuda_ut_file = '{}build/compile_commands.json'.format(PADDLE_ROOT)
        with open(cpp_cuda_ut_file) as f:
            data = json.load(f)
        for ut in data:
            ut_name = re.search('-o .*\/(.*).dir\/.*', ut['command']).group(1)
            if ut_name not in self.ut_list:
                continue
            self.src_ut_dict[ut['file']] = ut_name

    def load_python_ut(self):
        """ Load Python unit test list. """
        pyut_files = subprocess.check_output(
            'find {}python -name test*.py'.format(PADDLE_ROOT).split(' '))
        pyut_list = pyut_files.rstrip('\n').split('\n')
        for src_file in pyut_list:
            self.src_ut_dict[src_file] = src_file.split('/')[-1].split('.py')[0]

    def get_src_ut_map(self):
        """ Get src file and unit test map. """
        self.load_ctest_ut_list()
        self.load_cpp_and_cuda_ut()
        self.load_python_ut()
        return self.src_ut_dict


if __name__ == '__main__':
    ut_mapper = UTMapper()
    ut_mapper.load_python_ut()
    ut_mapper.load_ctest_ut_list()
