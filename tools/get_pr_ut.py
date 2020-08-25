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
""" For the PR that only modified the unit test, get cases in pull request. """

import os
from github import Github
import ut_filter

PADDLE_ROOT = os.getenv('PADDLE_ROOT', '/paddle/')


class PRChecker(object):
    """ PR Checker. """

    def __init__(self):
        self.github = Github(os.getenv('GITHUB_API_TOKEN'), timeout=60)
        self.repo = self.github.get_repo('PaddlePaddle/Paddle')
        self.pr = None

    def init(self):
        """ Get pull request. """
        pr_id = os.getenv('GIT_PR_ID')
        if not pr_id:
            print('No PR ID')
            exit(0)
        self.pr = self.repo.get_pull(int(pr_id))

    def get_pr_files(self):
        """ Get files in pull request. """
        page = 0
        file_list = []
        while True:
            files = self.pr.get_files().get_page(page)
            if not files:
                break
            for f in files:
                file_list.append(PADDLE_ROOT + f.filename)
            page += 1
        return file_list
        #return ['/paddle/paddle/fluid/memory/malloc_test.cu']

    def get_pr_ut(self):
        """ Get unit tests in pull request. """
        ut_str = ''
        ut_mapper = ut_filter.UTMapper()
        file_ut_map = ut_mapper.get_src_ut_map()
        for f in self.get_pr_files():
            if f not in file_ut_map:
                return ''
            else:
                ut_str = '{}^{}$|'.format(ut_str, file_ut_map[f])
        return ut_str.rstrip('|')


if __name__ == '__main__':
    pr_checker = PRChecker()
    pr_checker.init()
    print(pr_checker.get_pr_ut())
