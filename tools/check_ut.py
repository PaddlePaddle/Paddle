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
""" Get pull requests. """

import os
import os.path
from github import Github


class PRChecker:
    """PR Checker."""

    def __init__(self):
        self.github = Github(os.getenv('GITHUB_API_TOKEN'), timeout=60)
        self.repo = None

    def check(self, filename, msg):
        """
        Args:
            filename (str): File to get block names.
            msg (str): Error message.
        """
        pr_id = os.getenv('GIT_PR_ID')
        if not pr_id:
            print('No PR ID')
            exit(0)
        print(pr_id)
        if not os.path.isfile(filename):
            print('No author to check')
            exit(0)
        self.repo = self.github.get_repo('PaddlePaddle/Paddle')
        pr = self.repo.get_pull(int(pr_id))
        user = pr.user.login
        with open(filename) as f:
            for l in f:
                if l.rstrip('\r\n') == user:
                    print('{} {}'.format(user, msg))


if __name__ == '__main__':
    pr_checker = PRChecker()
    pr_checker.check('block.txt', 'has unit-test to be fixed, so CI failed.')
    pr_checker.check('bk.txt', 'has benchmark issue to be fixed, so CI failed.')
