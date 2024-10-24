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
import ssl
import sys

from github import Github

PADDLE_ROOT = os.getenv('PADDLE_ROOT', '/paddle/')
PADDLE_ROOT += '/'
PADDLE_ROOT = PADDLE_ROOT.replace('//', '/')
ssl._create_default_https_context = ssl._create_unverified_context


class PRChecker:
    """PR Checker."""

    def __init__(self):
        self.github = Github(os.getenv('GITHUB_API_TOKEN'), timeout=60)
        self.repo = self.github.get_repo('PaddlePaddle/Paddle')
        self.pr = None

    def init(self):
        """Get pull request."""
        pr_id = os.getenv('GIT_PR_ID')
        if not pr_id:
            print('PREC No PR ID')
            sys.exit(0)
        self.pr = self.repo.get_pull(int(pr_id))

    def get_pr_files(self):
        """Get files in pull request."""
        page = 0
        file_dict = {}
        while True:
            files = self.pr.get_files().get_page(page)
            if not files:
                break
            for f in files:
                file_dict[PADDLE_ROOT + f.filename] = f.status
            page += 1
        print(f"pr modify files: {file_dict}")
        return file_dict

    def check_only_change_python_file(self):
        file_dict = self.get_pr_files()
        for filename in file_dict:
            if not (
                filename.startswith(PADDLE_ROOT + 'python/')
                and filename.endswith('.py')
            ):
                return False
        return True


if __name__ == '__main__':
    pr_checker = PRChecker()
    pr_checker.init()
    if pr_checker.check_only_change_python_file():
        with open('only_change_python_file.txt', 'w') as f:
            f.write('yes')
