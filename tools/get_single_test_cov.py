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

import json
import os
import re
import subprocess
import sys


def getFNDAFile(rootPath, test):
    # load base fnda
    fnda_base_dict = {}
    find_file_cmd = os.popen("find %s -name %s.cc" % (rootPath, test))
    if find_file_cmd.read() != "":
        print("%s is a c++ unittest" % test)
        with open(
            "%s/build/ut_map/simple_precision_test/base_fnda.json" % rootPath,
            'r',
        ) as load_f:
            fnda_base_dict = json.load(load_f)
    # analyse fnda
    filename = '%s/build/ut_map/%s/coverage.info.tmp' % (rootPath, test)
    fn_filename = '%s/build/ut_map/%s/fnda.tmp' % (rootPath, test)
    os.system('touch %s' % fn_filename)
    try:
        f = open(filename)
        print("oepn %s succesfully" % filename)
    except FileNotFoundError:
        print("%s is not found." % filename)
        return
    all_data = f.read().split('TN:')
    del all_data[0]
    for gcov_data in all_data:
        message_list = gcov_data.split('\n')
        os.system('echo %s >> %s' % (message_list[1], fn_filename))
        if 'FNH:0' not in gcov_data:
            for message in message_list:
                if message.startswith(('FNDA:')) and (
                    not message.startswith(('FNDA:0,'))
                ):
                    tmp_data = message.split('FNDA:')[1].split(',')
                    hit = int(tmp_data[0])
                    symbol = tmp_data[1]
                    if symbol in fnda_base_dict:
                        if (hit - fnda_base_dict[symbol]) > 0:
                            fnda_str = 'FNDA:%s,%s' % (
                                str(hit - fnda_base_dict[symbol]),
                                symbol,
                            )
                            os.system('echo %s >> %s' % (fnda_str, fn_filename))
                    else:
                        os.system('echo %s >> %s' % (message, fn_filename))
    f.close()


def analysisFNDAFile(rootPath, test):
    related_ut_map_file = '%s/build/ut_map/%s/related_%s.txt' % (
        rootPath,
        test,
        test,
    )
    notrelated_ut_map_file = '%s/build/ut_map/%s/notrelated_%s.txt' % (
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

    fn_filename = '%s/build/ut_map/%s/fnda.tmp' % (rootPath, test)
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
                os.system(
                    'echo %s >> %s' % (clazz_filename, related_ut_map_file)
                )
            else:
                os.system(
                    'echo %s >> %s' % (clazz_filename, notrelated_ut_map_file)
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


def getBaseFnda(rootPath, test):
    filename = '%s/build/ut_map/%s/coverage.info.tmp' % (rootPath, test)
    try:
        f = open(filename)
        print("oepn %s succesfully" % filename)
    except FileNotFoundError:
        print("%s is not found." % filename)
    symbol_fnda = {}
    all_data = f.read().split('TN:')
    del all_data[0]
    for gcov_data in all_data:
        message_list = gcov_data.split('\n')
        # only for cc file
        if ".cc" in message_list[1]:
            for message in message_list:
                if message.startswith(('FNDA:')) and (
                    not message.startswith(('FNDA:0,'))
                ):
                    tmp_data = message.split('FNDA:')[1].split(',')
                    symbol_fnda[tmp_data[1]] = int(tmp_data[0])
    f.close()

    with open("%s/build/ut_map/%s/base_fnda.json" % (rootPath, test), "w") as f:
        json.dump(symbol_fnda, f, indent=4)


def getCovinfo(rootPath, test):
    ut_map_path = '%s/build/ut_map/%s' % (rootPath, test)
    print("start get fluid ===>")
    cmd_fluid = (
        'cd %s lcov --capture -d ./paddle/fluid/ -o ./paddle/fluid/coverage_fluid.info --rc lcov_branch_coverage=0'
        % ut_map_path
    )
    p_fluid = subprocess.Popen(cmd_fluid, shell=True, stdout=subprocess.DEVNULL)

    print("start get phi ===>")
    cmd_phi = (
        'cd %s lcov --capture -d ./paddle/phi -o ./paddle/phi/coverage_phi.info --rc lcov_branch_coverage=0'
        % ut_map_path
    )
    p_phi = subprocess.Popen(cmd_phi, shell=True, stdout=subprocess.DEVNULL)

    print("start get utils ===>")
    cmd_utils = (
        'cd %s lcov --capture -d ./paddle/utils -o ./paddle/utils/coverage_utils.info --rc lcov_branch_coverage=0'
        % ut_map_path
    )
    p_utils = subprocess.Popen(cmd_utils, shell=True, stdout=subprocess.DEVNULL)

    print("start wait fluid ===>")
    p_fluid.wait()
    print("start wait phi ===>")
    p_phi.wait()
    print("start wait utils ===>")
    p_utils.wait()
    print("end wait...")
    os.system(
        'cd %s && lcov -a paddle/fluid/coverage_fluid.info -a paddle/phi/coverage_phi.info -a paddle/utils/coverage_utils.info -o coverage.info --rc lcov_branch_coverage=0 > /dev/null 2>&1'
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
    if test == "simple_precision_test":
        getBaseFnda(rootPath, test)
    else:
        getFNDAFile(rootPath, test)
        analysisFNDAFile(rootPath, test)
    os.system('rm -rf %s/coverage.info.tmp' % ut_map_path)


if __name__ == "__main__":
    rootPath = sys.argv[1]
    case = sys.argv[2]
    getCovinfo(rootPath, case)
