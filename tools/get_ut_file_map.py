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
import sys


def get_all_paddle_file(rootPath):
    """get all file in Paddle repo: paddle/fluid, python"""
    traverse_files = [f'{rootPath}']
    all_file_paddle = f'{rootPath}/build/all_file_paddle'
    all_file_paddle_list = []
    with open(all_file_paddle, 'w') as f:
        for filename in traverse_files:
            g = os.walk(filename)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    all_file_paddle_list.append(os.path.join(path, file_name))
    return all_file_paddle_list


def get_all_uts(rootPath):
    all_uts_paddle = f'{rootPath}/build/all_uts_paddle'
    os.system(
        fr'cd {rootPath}/build && ctest -N -V | grep -Ei "Test[ \t]+#" | grep -oEi "\w+$" > {all_uts_paddle}'
    )


def remove_useless_file(rootPath):
    """remove useless file in ut_file_map.json"""
    all_file_paddle_list = get_all_paddle_file(rootPath)
    ut_file_map_new = {}
    ut_file_map = f"{rootPath}/build/ut_file_map.json"
    with open(ut_file_map, 'r') as load_f:
        load_dict = json.load(load_f)
    for key in load_dict:
        if key in all_file_paddle_list:
            ut_file_map_new[key] = load_dict[key]

    with open(f"{rootPath}/build/ut_file_map.json", "w") as f:
        json.dump(ut_file_map_new, f, indent=4)
        print("remove_useless_file ut_file_map success!!")


def handle_ut_file_map(rootPath):
    utNotSuccess_list = []
    ut_map_path = f"{rootPath}/build/ut_map"
    files = os.listdir(ut_map_path)
    ut_file_map = {}
    count = 0
    not_success_file = open(f"{rootPath}/build/prec_delta", 'w')
    # if testdir is not made,write the test into prec_delta
    get_all_uts(rootPath)
    all_ut = f'{rootPath}/build/all_uts_paddle'
    with open(all_ut, 'r') as f:
        all_ut_list = []
        for ut in f:
            ut = ut.replace('\n', '')
            all_ut_list.append(ut.strip())
        f.close()
    for ut in all_ut_list:
        filedir = f'{rootPath}/build/ut_map/{ut}'
        if not os.path.exists(filedir):
            not_success_file.write(f'{ut}\n')
            utNotSuccess_list.append(ut)
    # if fnda.tmp not exists,write the test into prec_delta
    for ut in files:
        count = count + 1
        print(f"ut {count}: {ut}")
        coverage_info = f'{ut_map_path}/{ut}/fnda.tmp'
        if os.path.exists(coverage_info):
            filename = f'{ut_map_path}/{ut}/related_{ut}.txt'
            try:
                f = open(filename)
                print(f"open {filename} successfully")
            except FileNotFoundError:
                print(f"{filename} is not found.")
                return
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').strip()
                if line == '':
                    continue
                elif line.startswith('/paddle/build'):
                    source_file = line.replace('/build', '')
                    # source_file = re.sub('.pb.*', '.proto', source_file)
                elif 'precise test map fileeee:' in line:
                    source_file = line.split('precise test map fileeee:')[
                        1
                    ].strip()
                else:
                    source_file = line
                if source_file not in ut_file_map:
                    ut_file_map[source_file] = []
                if ut not in ut_file_map[source_file]:
                    ut_file_map[source_file].append(ut)
            f.close()
        else:
            not_success_file.write(f'{ut}\n')
            utNotSuccess_list.append(ut)
    not_success_file.close()

    print("utNotSuccess:")
    print(utNotSuccess_list)

    for ut in files:
        if ut not in utNotSuccess_list:
            filename = f'{ut_map_path}/{ut}/notrelated_{ut}.txt'
            try:
                f = open(filename)
                print(f"open {filename} successfully")
            except FileNotFoundError:
                print(f"{filename} is not found.")
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').strip()
                if line == '':
                    continue
                elif line.startswith('/paddle/build'):
                    source_file = line.replace('/build', '')
                else:
                    source_file = line
                if source_file not in ut_file_map:
                    ut_file_map[source_file] = []
            f.close()
    with open(f"{rootPath}/build/ut_file_map.json", "w") as f:
        json.dump(ut_file_map, f, indent=4)


def notsuccessfuc(rootPath):
    utNotSuccess = ''
    ut_map_path = f"{rootPath}/build/ut_map"
    files = os.listdir(ut_map_path)
    count = 0

    # ut failed!!
    for ut in files:
        if ut == 'simple_precise_test':
            continue
        coverage_info = f'{ut_map_path}/{ut}/fnda.tmp'
        if os.path.exists(coverage_info):
            pass
        else:
            count = count + 1
            utNotSuccess = utNotSuccess + f'^{ut}$|'

    # ut not exec

    get_all_uts(rootPath)
    with open("/paddle/build/all_uts_paddle", "r") as f:
        data = f.readlines()
    for ut in data:
        ut = ut.replace('\n', '').strip()
        if ut not in files:
            print(ut)
            count = count + 1
            utNotSuccess = utNotSuccess + f'^{ut}$|'

    if utNotSuccess != '':
        print(f"utNotSuccess count: {count}")
        f = open(f'{rootPath}/build/utNotSuccess', 'w')
        f.write(utNotSuccess[:-1])
        f.close()


def ut_file_map_supplement(rootPath):
    ut_file_map_new = f"{rootPath}/build/ut_file_map.json"
    precision_test_map_store_dir = "/precision_test_map_store"
    os.system(f'mkdir {precision_test_map_store_dir}')
    os.system(
        f'cd {precision_test_map_store_dir} && wget --no-proxy https://paddle-docker-tar.bj.bcebos.com/tmp_test/ut_file_map.json --no-check-certificate'
    )
    ut_file_map_old = f"{precision_test_map_store_dir}/ut_file_map.json"
    with open(ut_file_map_new, 'r') as load_f:
        load_dict_new = json.load(load_f)

    all_uts_paddle = f'{rootPath}/build/all_uts_paddle'

    with open(all_uts_paddle, 'r') as f:
        all_uts_paddle_list = []
        for ut in f:
            all_uts_paddle_list.append(ut.strip())
        f.close()

    with open(f"{precision_test_map_store_dir}/ut_file_map.json", "w") as f:
        json.dump(load_dict_new, f, indent=4)
        print("load_dict_new success!!")

    os.system(
        f'cd {precision_test_map_store_dir} && wget --no-proxy https://paddle-docker-tar.bj.bcebos.com/tmp_test/prec_delta --no-check-certificate'
    )
    prec_delta_new = f"{rootPath}/build/prec_delta"
    with open(prec_delta_new, 'r') as f:
        prec_delta_new_list = []
        for ut in f:
            prec_delta_new_list.append(ut.strip())
        f.close()
    prec_delta_new_list.append(
        'test_py_reader_error_msg'
    )  # add a python case for pycoverage
    prec_delta_file = open(f"{precision_test_map_store_dir}/prec_delta", 'w')
    for ut in prec_delta_new_list:
        prec_delta_file.write(ut + '\n')
    print("prec_delta_file success!!")
    prec_delta_file.close()


def utmap_analysis(rootPath):
    ut_file_map_new = f"{rootPath}/build/ut_file_map.json"
    with open(ut_file_map_new, 'r') as load_f:
        load_dict_new = json.load(load_f)
    print(len(load_dict_new))
    for filename in load_dict_new:
        print(filename, len(load_dict_new[filename]))


if __name__ == "__main__":
    func = sys.argv[1]
    if func == 'get_not_success_ut':
        rootPath = sys.argv[2]
        notsuccessfuc(rootPath)
    elif func == 'get_ut_map':
        rootPath = sys.argv[2]
        handle_ut_file_map(rootPath)
        remove_useless_file(rootPath)
        ut_file_map_supplement(rootPath)
