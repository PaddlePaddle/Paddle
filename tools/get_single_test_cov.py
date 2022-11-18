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

import os
import sys
import re


def getFNDAFile(rootPath, test):
    filename = f'{rootPath}/build/ut_map/{test}/coverage.info.tmp'
    fn_filename = f'{rootPath}/build/ut_map/{test}/fnda.tmp'
    os.system('touch %s' % fn_filename)
    try:
        f = open(filename)
        print("oepn %s succesfully" % filename)
    except FileNotFoundError:
        print("%s is not found." % filename)
        return
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '')
        if line.startswith('SF:'):
            os.system(f'echo {line} >> {fn_filename}')
        elif line.startswith('FNDA:'):
            hit = int(line.split('FNDA:')[1].split(',')[0])
            if hit != 0:
                os.system(f'echo {line} >> {fn_filename}')
    f.close()


def analysisFNDAFile(rootPath, test):
    related_ut_map_file = '{}/build/ut_map/{}/related_{}.txt'.format(
        rootPath,
        test,
        test,
    )
    notrelated_ut_map_file = '{}/build/ut_map/{}/notrelated_{}.txt'.format(
        rootPath,
        test,
        test,
    )
    os.system('touch %s' % related_ut_map_file)
    os.system('touch %s' % notrelated_ut_map_file)

    if os.path.isfile(related_ut_map_file) and os.path.isfile(
        notrelated_ut_map_file
    ):
        print("make related.txt and not_related.txt succesfully")
    else:
        print("make related.txt and not_related.txt failed")
        return

    fn_filename = f'{rootPath}/build/ut_map/{test}/fnda.tmp'
    try:
        f = open(fn_filename)
        print("oepn %s succesfully" % fn_filename)
    except FileNotFoundError:
        print("%s is not found." % fn_filename)
        return
    data = f.read().split('SF:')
    related_file_list = []
    for message in data:
        message_list = message.split('\n')
        clazz_filename = message_list[0]
        if '/build/' in clazz_filename:
            clazz_filename = clazz_filename.replace('/build', '')
        if '.pb.h' in clazz_filename:
            clazz_filename = clazz_filename.replace('.pb.h', '.proto')
        if '.pb.cc' in clazz_filename:
            clazz_filename = clazz_filename.replace('.pb.cc', '.proto')
        if 'FNDA:' in message:
            OP_REGIST = True
            for i in range(1, len(message_list) - 1):
                fn = message_list[i]
                matchObj = re.match(
                    r'(.*)Maker(.*)|(.*)Touch(.*)Regist(.*)|(.*)Touch(.*)JitKernel(.*)|(.*)converterC2Ev(.*)',
                    fn,
                    re.I,
                )
                if matchObj is None:
                    OP_REGIST = False
                    break
            if not OP_REGIST:
                related_file_list.append(clazz_filename)
                os.system(f'echo {clazz_filename} >> {related_ut_map_file}')
            else:
                os.system(
                    'echo {} >> {}'.format(
                        clazz_filename, notrelated_ut_map_file
                    )
                )
        else:
            if clazz_filename != '':
                if (
                    clazz_filename not in related_file_list
                ):  # xx.pb.cc in RELATED xx.pb.h not in RELATED
                    os.system(
                        'echo %s >> %s'
                        % (clazz_filename, notrelated_ut_map_file)
                    )
    f.close()


def getCovinfo(rootPath, test):
    ut_map_path = f'{rootPath}/build/ut_map/{test}'
    os.system(
        'cd %s && lcov --capture -d . -o coverage.info --rc lcov_branch_coverage=0 > /dev/null 2>&1'
        % ut_map_path
    )
    coverage_info_path = ut_map_path + '/coverage.info'
    file_size = os.path.getsize(coverage_info_path)
    if file_size == 0:
        print("coverage.info is empty,collect coverage rate failed")
        return
    else:
        print("get coverage.info succesfully")
    os.system(
        "cd %s && lcov --extract coverage.info '/paddle/paddle/phi/*' '/paddle/paddle/utils/*' '/paddle/paddle/fluid/framework/*' '/paddle/paddle/fluid/imperative/*' '/paddle/paddle/fluid/inference/*' '/paddle/paddle/fluid/memory/*' '/paddle/paddle/fluid/operators/*' '/paddle/paddle/fluid/string/*' '/paddle/paddle/fluid/distributed/*' '/paddle/paddle/fluid/platform/*' '/paddle/paddle/fluid/pybind/*' '/paddle/build/*' -o coverage.info.tmp --rc lcov_branch_coverage=0 > /dev/null 2>&1"
        % ut_map_path
    )
    coverage_info_tmp = ut_map_path + '/coverage.info.tmp'
    coverage_tmp_size = os.path.getsize(coverage_info_tmp)
    if coverage_tmp_size == 0:
        print("coverage.info.tmp is empty,collect coverage rate failed")
        return
    else:
        print("get coverage.info.tmp succesfully")

    os.system('rm -rf %s/paddle' % ut_map_path)
    os.system('rm -rf %s/coverage.info' % ut_map_path)
    getFNDAFile(rootPath, test)
    analysisFNDAFile(rootPath, test)
    os.system('rm -rf %s/coverage.info.tmp' % ut_map_path)


if __name__ == "__main__":
    rootPath = sys.argv[1]
    case = sys.argv[2]
    getCovinfo(rootPath, case)
