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

import json
import os
import platform
import re
import ssl
import subprocess
import time
import urllib.request

import requests
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
        self.py_prog_oneline = re.compile(r'\d+\|\s*#.*')
        self.py_prog_multiline_a = re.compile('"""(.*?)"""', re.DOTALL)
        self.py_prog_multiline_b = re.compile("'''(.*?)'''", re.DOTALL)
        self.cc_prog_online = re.compile(r'\d+\|\s*//.*')
        self.cc_prog_multiline = re.compile(r'\d+\|\s*/\*.*?\*/', re.DOTALL)
        self.lineno_prog = re.compile(r'@@ \-\d+,\d+ \+(\d+),(\d+) @@')
        self.pr = None
        self.suffix = ''
        self.full_case = False

    def init(self):
        """Get pull request."""
        pr_id = os.getenv('GIT_PR_ID')
        if not pr_id:
            print('PREC No PR ID')
            exit(0)
        suffix = os.getenv('PREC_SUFFIX')
        if suffix:
            self.suffix = suffix
        self.pr = self.repo.get_pull(int(pr_id))
        last_commit = None
        ix = 0
        while True:
            try:
                commits = self.pr.get_commits().get_page(ix)
                if len(commits) == 0:
                    raise ValueError("no commit found in {} page".format(ix))
                last_commit = commits[-1].commit
            except Exception as e:
                break
            else:
                ix = ix + 1
        if last_commit.message.find('test=allcase') != -1:
            print('PREC test=allcase is set')
            self.full_case = True

    # todo: exception
    def __wget_with_retry(self, url):
        ix = 1
        proxy = '--no-proxy'
        while ix < 6:
            if ix // 2 == 0:
                proxy = ''
            else:
                if platform.system() == 'Windows':
                    proxy = '-Y off'
                else:
                    proxy = '--no-proxy'
            code = subprocess.call(
                'wget -q {} --no-check-certificate {}'.format(proxy, url),
                shell=True,
            )
            if code == 0:
                return True
            print(
                'PREC download {} error, retry {} time(s) after {} secs.[proxy_option={}]'.format(
                    url, ix, ix * 10, proxy
                )
            )
            time.sleep(ix * 10)
            ix += 1
        return False

    def __urlretrieve(self, url, filename):
        ix = 1
        with_proxy = urllib.request.getproxies()
        without_proxy = {'http': '', 'http': ''}
        while ix < 6:
            if ix // 2 == 0:
                cur_proxy = urllib.request.ProxyHandler(without_proxy)
            else:
                cur_proxy = urllib.request.ProxyHandler(with_proxy)
            opener = urllib.request.build_opener(
                cur_proxy, urllib.request.HTTPHandler
            )
            urllib.request.install_opener(opener)
            try:
                urllib.request.urlretrieve(url, filename)
            except Exception as e:
                print(e)
                print(
                    'PREC download {} error, retry {} time(s) after {} secs.[proxy_option={}]'.format(
                        url, ix, ix * 10, cur_proxy
                    )
                )
                continue
            else:
                return True
            time.sleep(ix * 10)
            ix += 1

        return False

    def get_pr_files(self):
        """Get files in pull request."""
        page = 0
        file_dict = {}
        file_count = 0
        while True:
            files = self.pr.get_files().get_page(page)
            if not files:
                break
            for f in files:
                file_dict[PADDLE_ROOT + f.filename] = f.status
                file_count += 1
            if file_count == 30:  # if pr file count = 31, nend to run all case
                break
            page += 1
        print("pr modify files: %s" % file_dict)
        return file_dict

    def get_is_white_file(self, filename):
        """judge is white file in pr's files."""
        isWhiteFile = False
        not_white_files = (
            PADDLE_ROOT + 'cmake/',
            PADDLE_ROOT + 'patches/',
            PADDLE_ROOT + 'tools/dockerfile/',
            PADDLE_ROOT + 'tools/windows/',
            PADDLE_ROOT + 'tools/test_runner.py',
            PADDLE_ROOT + 'tools/parallel_UT_rule.py',
        )
        if 'cmakelist' in filename.lower():
            isWhiteFile = False
        elif filename.startswith((not_white_files)):
            isWhiteFile = False
        else:
            isWhiteFile = True
        return isWhiteFile

    def __get_comment_by_filetype(self, content, filetype):
        result = []
        if filetype == 'py':
            result = self.__get_comment_by_prog(content, self.py_prog_oneline)
            result.extend(
                self.__get_comment_by_prog(content, self.py_prog_multiline_a)
            )
            result.extend(
                self.__get_comment_by_prog(content, self.py_prog_multiline_b)
            )
        if filetype == 'cc':
            result = self.__get_comment_by_prog(content, self.cc_prog_oneline)
            result.extend(
                self.__get_comment_by_prog(content, self.cc_prog_multiline)
            )
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
        # content = self.repo.get_contents(f.replace(PADDLE_ROOT, ''), 'pull/').decoded_content
        # todo: get file from github
        with open(f, encoding="utf-8") as fd:
            lines = fd.readlines()
        lineno = 1
        inputs = ''
        for line in lines:
            # for line in content.split('\n'):
            # input += str(lineno) + '|' + line + '\n'
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
                            line = '{}{}'.format(
                                lineno, data[ix].replace('+', '|', 1)
                            )
                            if line_list:
                                line_list.append(line)
                            else:
                                file_to_diff_lines[filename] = [
                                    line,
                                ]
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
        print('PREC {} is only comment'.format(f))
        return True

    def get_all_count(self):
        p = subprocess.Popen(
            "cd {}build && ctest -N".format(PADDLE_ROOT),
            shell=True,
            stdout=subprocess.PIPE,
        )
        out, err = p.communicate()
        for line in out.splitlines():
            if 'Total Tests:' in str(line):
                all_counts = line.split()[-1]
        return int(all_counts)

    def file_is_unnit_test(self, unittest_path):
        # get all testcases by ctest-N
        all_ut_file = PADDLE_ROOT + 'build/all_ut_list'
        # all_ut_file = '%s/build/all_ut_file' % PADDLE_ROOT
        print("PADDLE_ROOT:", PADDLE_ROOT)
        print("all_ut_file path:", all_ut_file)
        build_path = PADDLE_ROOT + 'build/'
        print("build_path:", build_path)
        (unittest_directory, unittest_name) = os.path.split(unittest_path)
        # determine whether filename is in all_ut_case
        with open(all_ut_file, 'r') as f:
            all_unittests = f.readlines()
            for test in all_unittests:
                test = test.replace('\n', '').strip()
                if test == unittest_name.split(".")[0]:
                    return True
        return False

    def get_pr_ut(self):
        """Get unit tests in pull request."""
        if self.full_case:
            return ''
        check_added_ut = False
        ut_list = []
        file_ut_map = None

        ret = self.__urlretrieve(
            'https://paddle-docker-tar.bj.bcebos.com/tmp_test/ut_file_map.json',
            'ut_file_map.json',
        )
        if not ret:
            print('PREC download file_ut.json failed')
            exit(1)

        with open('ut_file_map.json') as jsonfile:
            file_ut_map = json.load(jsonfile)

        current_system = platform.system()
        notHitMapFiles = []
        hitMapFiles = {}
        onlyCommentsFilesOrXpu = []
        filterFiles = []
        file_list = []
        file_dict = self.get_pr_files()
        if len(file_dict) == 30:  # if pr file count = 31, nend to run all case
            return ''
        for filename in file_dict:
            if filename.startswith(PADDLE_ROOT + 'python/'):
                file_list.append(filename)
            elif filename.startswith(PADDLE_ROOT + 'paddle/'):
                if filename.startswith((PADDLE_ROOT + 'paddle/infrt')):
                    filterFiles.append(filename)
                elif filename.startswith(PADDLE_ROOT + 'paddle/scripts'):
                    if filename.startswith(
                        (
                            PADDLE_ROOT + 'paddle/scripts/paddle_build.sh',
                            PADDLE_ROOT + 'paddle/scripts/paddle_build.bat',
                        )
                    ):
                        file_list.append(filename)
                    else:
                        filterFiles.append(filename)
                elif (
                    ('/xpu/' in filename.lower())
                    or ('/npu/' in filename.lower())
                    or ('/mlu/' in filename.lower())
                    or ('/ipu/' in filename.lower())
                ):
                    filterFiles.append(filename)
                else:
                    file_list.append(filename)
            else:
                if file_dict[filename] == 'added':
                    file_list.append(filename)
                else:
                    isWhiteFile = self.get_is_white_file(filename)
                    if not isWhiteFile:
                        file_list.append(filename)
                    else:
                        filterFiles.append(filename)
        if len(file_list) == 0:
            ut_list.append('filterfiles_placeholder')
            ret = self.__urlretrieve(
                'https://paddle-docker-tar.bj.bcebos.com/tmp_test/prec_delta',
                'prec_delta',
            )
            if ret:
                with open('prec_delta') as delta:
                    for ut in delta:
                        ut_list.append(ut.rstrip('\r\n'))
            else:
                print('PREC download prec_delta failed')
                exit(1)
            PRECISION_TEST_Cases_ratio = format(
                float(len(ut_list)) / float(self.get_all_count()), '.2f'
            )
            print("filterFiles: %s" % filterFiles)
            print("ipipe_log_param_PRECISION_TEST: true")
            print(
                "ipipe_log_param_PRECISION_TEST_Cases_count: %s" % len(ut_list)
            )
            print(
                "ipipe_log_param_PRECISION_TEST_Cases_ratio: %s"
                % PRECISION_TEST_Cases_ratio
            )
            print(
                "The unittests in prec delta is shown as following: %s"
                % ut_list
            )
            return '\n'.join(ut_list)
        else:
            for f in file_list:
                if (
                    current_system == "Darwin"
                    or current_system == "Windows"
                    or self.suffix == ".py3"
                ):
                    f_judge = f.replace(PADDLE_ROOT, '/paddle/', 1)
                    f_judge = f_judge.replace('//', '/')
                else:
                    f_judge = f
                if f_judge not in file_ut_map:
                    if f_judge.endswith('.md'):
                        ut_list.append('md_placeholder')
                        onlyCommentsFilesOrXpu.append(f_judge)
                    elif (
                        'tests/unittests/xpu' in f_judge
                        or 'tests/unittests/npu' in f_judge
                        or 'op_npu.cc' in f_judge
                    ):
                        ut_list.append('xpu_npu_placeholder')
                        onlyCommentsFilesOrXpu.append(f_judge)
                    elif f_judge.endswith(('.h', '.cu', '.cc', '.py')):
                        # determine whether the new added file is a member of added_ut
                        if file_dict[f] in ['added']:
                            f_judge_in_added_ut = False
                            path = PADDLE_ROOT + 'added_ut'
                            print("PADDLE_ROOT:", PADDLE_ROOT)
                            print("adde_ut path:", path)
                            (unittest_directory, unittest_name) = os.path.split(
                                f_judge
                            )
                            with open(path, 'r') as f:
                                added_unittests = f.readlines()
                                for test in added_unittests:
                                    test = test.replace('\n', '').strip()
                                    if test == unittest_name.split(".")[0]:
                                        f_judge_in_added_ut = True
                            if f_judge_in_added_ut:
                                print(
                                    "Adding new unit tests not hit mapFiles: %s"
                                    % f_judge
                                )
                            else:
                                notHitMapFiles.append(f_judge)
                        elif file_dict[f] in ['removed']:
                            print("remove file not hit mapFiles: %s" % f_judge)
                        else:
                            if self.is_only_comment(f):
                                ut_list.append('comment_placeholder')
                                onlyCommentsFilesOrXpu.append(f_judge)
                            if self.file_is_unnit_test(f_judge):
                                ut_list.append(f_judge.split(".")[0])
                            else:
                                notHitMapFiles.append(f_judge)
                    else:
                        notHitMapFiles.append(f_judge) if file_dict[
                            f
                        ] != 'removed' else print(
                            "remove file not hit mapFiles: %s" % f_judge
                        )
                else:
                    if file_dict[f] not in ['removed']:
                        if self.is_only_comment(f):
                            ut_list.append('comment_placeholder')
                            onlyCommentsFilesOrXpu.append(f_judge)
                        else:
                            hitMapFiles[f_judge] = len(file_ut_map[f_judge])
                            ut_list.extend(file_ut_map.get(f_judge))
                    else:
                        hitMapFiles[f_judge] = len(file_ut_map[f_judge])
                        ut_list.extend(file_ut_map.get(f_judge))

            ut_list = list(set(ut_list))
            if len(notHitMapFiles) != 0:
                print("ipipe_log_param_PRECISION_TEST: false")
                print("notHitMapFiles: %s" % notHitMapFiles)
                if len(filterFiles) != 0:
                    print("filterFiles: %s" % filterFiles)
                return ''
            else:
                if ut_list:
                    ret = self.__urlretrieve(
                        'https://paddle-docker-tar.bj.bcebos.com/tmp_test/prec_delta',
                        'prec_delta',
                    )
                    if ret:
                        with open('prec_delta') as delta:
                            for ut in delta:
                                if ut not in ut_list:
                                    ut_list.append(ut.rstrip('\r\n'))
                    else:
                        print('PREC download prec_delta failed')
                        exit(1)
                    print("hitMapFiles: %s" % hitMapFiles)
                    print("ipipe_log_param_PRECISION_TEST: true")
                    print(
                        "ipipe_log_param_PRECISION_TEST_Cases_count: %s"
                        % len(ut_list)
                    )
                    PRECISION_TEST_Cases_ratio = format(
                        float(len(ut_list)) / float(self.get_all_count()), '.2f'
                    )
                    print(
                        "ipipe_log_param_PRECISION_TEST_Cases_ratio: %s"
                        % PRECISION_TEST_Cases_ratio
                    )
                    if len(filterFiles) != 0:
                        print("filterFiles: %s" % filterFiles)
                return '\n'.join(ut_list)


if __name__ == '__main__':
    pr_checker = PRChecker()
    pr_checker.init()
    with open('ut_list', 'w') as f:
        f.write(pr_checker.get_pr_ut())
