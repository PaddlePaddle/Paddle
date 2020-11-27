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
import json
from github import Github

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

    def get_pr_ut(self):
        """ Get unit tests in pull request. """
        check_added_ut = False
        ut_list = []
        file_ut_map = None
        cmd = 'wget -q --no-check-certificate https://sys-p0.bj.bcebos.com/prec/file_ut.json'
        os.system(cmd)
        with open('file_ut.json') as jsonfile:
            file_ut_map = json.load(jsonfile)
        for f in self.get_pr_files():
            if f.endswith('.h') or f.endswith('.cu'):
                return ''
            if f not in file_ut_map:
                if f.find('test_') != -1 or f.find('_test') != -1:
                    check_added_ut = True
                    continue
                else:
                    return ''
            else:
                ut_list.extend(file_ut_map.get(f))
        ut_list = list(set(ut_list))
        cmd = 'wget -q --no-check-certificate https://sys-p0.bj.bcebos.com/prec/prec_delta'
        os.system(cmd)
        with open('prec_delta') as delta:
            for ut in delta:
                ut_list.append(ut.rstrip('\r\n'))

        if check_added_ut:
            cmd = 'bash {}/tools/check_added_ut.sh >/tmp/pre_ut 2>&1'.format(
                PADDLE_ROOT)
            os.system(cmd)
            with open('{}/added_ut'.format(PADDLE_ROOT)) as utfile:
                for ut in utfile:
                    ut_list.append(ut.rstrip('\r\n'))

        return ' '.join(ut_list)


if __name__ == '__main__':
    pr_checker = PRChecker()
    pr_checker.init()
    print(pr_checker.get_pr_ut())
