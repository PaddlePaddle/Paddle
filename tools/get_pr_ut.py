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
import re
import sys
import requests
from github import Github

PADDLE_ROOT = os.getenv('PADDLE_ROOT', '/paddle/')
PADDLE_ROOT += '/'
PADDLE_ROOT = PADDLE_ROOT.replace('//', '/')


class PRChecker(object):
    """ PR Checker. """

    def __init__(self):
        self.github = Github(os.getenv('GITHUB_API_TOKEN'), timeout=60)
        self.repo = self.github.get_repo('PaddlePaddle/Paddle')
        self.py_prog_oneline = re.compile('\d+\|\s*#.*')
        self.py_prog_multiline_a = re.compile('\d+\|\s*r?""".*?"""', re.DOTALL)
        self.py_prog_multiline_b = re.compile("\d+\|\s*r?'''.*?'''", re.DOTALL)
        self.cc_prog_online = re.compile('\d+\|\s*//.*')
        self.cc_prog_multiline = re.compile('\d+\|\s*/\*.*?\*/', re.DOTALL)
        self.lineno_prog = re.compile('@@ \-\d+,\d+ \+(\d+),(\d+) @@')
        self.pr = None
        self.suffix = ''
        self.full_case = False

    def init(self):
        """ Get pull request. """
        pr_id = os.getenv('GIT_PR_ID')
        if not pr_id:
            print('No PR ID')
            exit(0)
        suffix = os.getenv('PREC_SUFFIX')
        if suffix:
            self.suffix = suffix
        self.pr = self.repo.get_pull(int(pr_id))
        last_commit = None
        ix = 0
        while True:
            commits = self.pr.get_commits().get_page(ix)
            for c in commits:
                last_commit = c.commit
            else:
                break
            ix = ix + 1
        if last_commit.message.find('test=full_case') != -1:
            self.full_case = True

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

    def __get_comment_by_filetype(self, content, filetype):
        result = []
        if filetype == 'py':
            result = self.__get_comment_by_prog(content, self.py_prog_oneline)
            result.extend(
                self.__get_comment_by_prog(content, self.py_prog_multiline_a))
            result.extend(
                self.__get_comment_by_prog(content, self.py_prog_multiline_b))
        if filetype == 'cc':
            result = self.__get_comment_by_prog(content, self.cc_prog_oneline)
            result.extend(
                self.__get_comment_by_prog(content, self.cc_prog_multiline))
        return result

    def __get_comment_by_prog(self, content, prog):
        result_list = prog.findall(content)
        if not result_list:
            return []
        result = []
        for u in result_list:
            result.extend(u.split('\n'))
        return result

    def get_comment_of_file(self, f):
        #content = self.repo.get_contents(f.replace(PADDLE_ROOT, ''), 'pull/').decoded_content
        with open(f) as fd:
            lines = fd.readlines()
        lineno = 1
        inputs = ''
        for line in lines:
            #for line in content.split('\n'):
            #input += str(lineno) + '|' + line + '\n'
            inputs += str(lineno) + '|' + line
            lineno += 1
        fietype = ''
        if f.endswith('.h') or f.endswith('.cc') or f.endswith('.cu'):
            filetype = 'cc'
        if f.endswith('.py'):
            filetype = 'py'
        else:
            return []
        return self.__get_comment_by_filetype(inputs, filetype)

    def get_pr_diff_lines(self):
        file_to_diff_lines = {}
        r = requests.get(self.pr.diff_url)
        data = r.text
        data = data.split('\n')
        ix = 0
        while ix < len(data):
            if data[ix].startswith('+++'):
                if data[ix].rstrip('\r\n') == '+++ /dev/null':
                    ix += 1
                    continue
                filename = data[ix][6:]
                ix += 1
                while ix < len(data):
                    result = self.lineno_prog.match(data[ix])
                    if not result:
                        break
                    lineno = int(result.group(1))
                    length = int(result.group(2))
                    ix += 1
                    end = ix + length
                    while ix < end:
                        if data[ix][0] == '-':
                            end += 1
                        if data[ix][0] == '+':
                            line_list = file_to_diff_lines.get(filename)
                            line = '{}{}'.format(lineno,
                                                 data[ix].replace('+', '|', 1))
                            if line_list:
                                line_list.append(line)
                            else:
                                file_to_diff_lines[filename] = [line, ]
                        if data[ix][0] != '-':
                            lineno += 1
                        ix += 1
            ix += 1
        return file_to_diff_lines

    def is_only_comment(self, f):
        file_to_diff_lines = self.get_pr_diff_lines()
        comment_lines = self.get_comment_of_file(f)
        diff_lines = file_to_diff_lines.get(f.replace(PADDLE_ROOT, '', 1))
        if not diff_lines:
            return False
        for l in diff_lines:
            if l not in comment_lines:
                return False
        return True

    def get_pr_ut(self):
        """ Get unit tests in pull request. """
        if self.full_case:
            return ''
        check_added_ut = False
        ut_list = []
        file_ut_map = None
        cmd = 'wget -q --no-proxy --no-check-certificate https://sys-p0.bj.bcebos.com/prec/file_ut.json' + self.suffix
        os.system(cmd)
        with open('file_ut.json' + self.suffix) as jsonfile:
            file_ut_map = json.load(jsonfile)
        for f in self.get_pr_files():
            if f not in file_ut_map:
                if f.endswith('.md'):
                    ut_list.append('md_placeholder')
                elif f.endswith('.h') or f.endswith('.cu'):
                    if self.is_only_comment(f):
                        ut_list.append('h_cu_comment_placeholder')
                    else:
                        return ''
                elif f.endswith('.cc') or f.endswith('.py') or f.endswith(
                        '.cu'):
                    if f.find('test_') != -1 or f.find('_test') != -1:
                        check_added_ut = True
                    elif self.is_only_comment(f):
                        ut_list.append('nomap_comment_placeholder')
                    else:
                        return ''
                else:
                    return ''
            else:
                if self.is_only_comment(f):
                    ut_list.append('map_comment_placeholder')
                else:
                    ut_list.extend(file_ut_map.get(f))
        ut_list = list(set(ut_list))
        cmd = 'wget -q --no-proxy --no-check-certificate https://sys-p0.bj.bcebos.com/prec/prec_delta' + self.suffix
        os.system(cmd)
        with open('prec_delta' + self.suffix) as delta:
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
