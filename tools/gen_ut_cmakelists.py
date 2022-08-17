# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import re


# function to process pythonpath env
# append "${PADDLE_BINARY_DIR}/python" to PYTHONPATH
def _process_PYTHONPATH(pythonpath_option):
    pythonpath_option += ":${PADDLE_BINARY_DIR}/python"
    return pythonpath_option


def process_envs(envs):
    """
    Desc:
        Input a str and output a str with the same function to specify some environment variables.
    Here we can give a specital process for some variable if needed.
    Example 1:
        Input: "http_proxy=;PYTHONPATH=.."
        Output: "http_proxy=;PYTHONPATH=..:${PADDLE_BINARY_DIR}/python"
    Example 2:
        Input: "http_proxy=;https_proxy=123.123.123.123:1230"
        Output: "http_proxy=;https_proxy=123.123.123.123:1230"
    """
    envs = envs.strip()

    envs_parts = envs.split(";")
    processed_envs = []

    for p in envs_parts:
        assert " " not in p and \
            re.compile("^[a-zA-Z_][0-9a-zA-Z_]*=").search(p) is not None, \
            f"""The environment option format is wrong. The env variable name can only contains'a-z', 'A-Z', '0-9' and '_',
and the var can not contain space in either env names or values.
However the var's format is '{p}'."""

        if re.compile("^PYTHONPATH=").search(p):
            p = _process_PYTHONPATH(p)

        processed_envs.append(p)

    return ";".join(processed_envs)


def process_conditions(conditions):
    """
    Desc:
        Input condition expression in cmake grammer and return a string warpped by 'AND ()'.
        If the conditions string is empty, return an empty string.
    Example 1:
        Input: "LINUX"
        Output: "AND (LINUX)"
    Example 2:
        Input: ""
        Output: ""
    """
    if len(conditions.strip()) == 0:
        conditions = ""
    else:
        conditions = f" AND ({conditions})"
    return conditions


def proccess_archs(arch):
    """
    desc:
        Input archs options and warp it with 'WITH_', 'OR' and '()' in cmakelist grammer.
        The case is ignored.
        If the input is empty, return "LOCAL_ALL_ARCH".
    Example 1:
        Input: 'gpu'
        Output: '(WITH_GPU)'
    Example 2:
        Input: 'gpu;ROCM'
        Output: '(WITH_GPU OR WITH_ROCM)'
    """
    archs = ""
    arch = arch.upper().strip()
    if len(arch) > 0:
        for a in arch.split(";"):
            assert a in ["GPU", "ROCM", "ASCEND", "ASCEND_CL"], \
                f"""Supported arhc options are "GPU", "ROCM", "ASCEND" and "ASCEND_CL", but the options is {a}"""
            archs += "WITH_" + a.upper() + " OR "
        arch = "(" + archs[:-4] + ")"
    else:
        arch = "LOCAL_ALL_ARCH"
    return arch


def process_os(os_):
    """
    Desc:
        Input os options and output warpped options with 'OR' and '()'
        If the input is empty, return "LOCAL_ALL_PLAT"
    Example 1:
        Input: "WIN32"
        Output: "(WIN32)"
    Example 2:
        Input: "WIN32;linux"
        Output: "(WIN32 OR LINUX)"
    """
    os_ = os_.strip()
    if len(os_) > 0:
        os_ = os_.upper()
        for p in os_.split(';'):
            assert p in [
                "WIN32", "APPLE", "LINUX"
            ], f"""Supported os options are 'WIN32', 'APPLE' and 'LINUX', but the options is {p}"""
        os_ = os_.replace(";", " OR ")
        os_ = "(" + os_ + ")"
    else:
        os_ = "LOCAL_ALL_PLAT"
    return os_


# check whether run_serial is 0, 1 or empty
def process_run_serial(run_serial):
    rs = run_serial.strip()
    assert rs in ["1", "0", ""], \
        f"""the value of run_serial must be one of 0, 1 or empty. But this value is {rs}"""
    if rs == "":
        rs = "0"
    return rs


def parse_line(line):
    """
    Desc:
        Input a line in csv file and output a string in cmake grammer, adding the specified test and setting its properties.
    Example:
        Input: "test_allreduce,linux,gpu;rocm,120,DIST,test_runner.py,20071,1,PYTHONPATH=..;http_proxy=;https_proxy=,"
        Output:
            "if((WITH_GPU OR WITH_ROCM) AND (LINUX) )
                py_test_modules(
                test_allreduce
                MODULES
                test_allreduce
                ENVS
                "PADDLE_DIST_UT_PORT=20071;PYTHONPATH=..:${PADDLE_BINARY_DIR}/python;http_proxy=;https_proxy=")
                set_tests_properties(test_allreduce PROPERTIES  TIMEOUT "120" RUN_SERIAL 1)
            endif()"
    """

    # A line contains name, os_, archs, timeout, run_type, launcher, dist_ut_port, run_serial, envs, conditions, etc.
    # Following are descriptions of each variable:
    #
    # * `name`: the test's name
    # * `os`: The supported operator system, ignoring case. If the test run in multiple operator systems, use ";" to split systems, forexample, `apple;linux` means the test runs on both Apple and Linux. The supported values are `linux`,`win32` and `apple`. If the value is empty, this means the test runs on all opertaor systems.
    # * `arch`: the device's architecture. similar to `os`, multiple valuse ars splited by ";" and ignoring case. The supported arhchetectures are `gpu`, `xpu`, `npu` and `rocm`.
    # * `timeout`: timeout of a unittest, whose unit is second.
    # * `run_type`: run_type of a unittest. Supported values are `NIGHTLY`, `EXCLUSIVE`, `CINN`, `DIST`, `GPUPS`, `INFER`，which are case-insensitive. Multiple Values are splited by ":".
    # * `launcher`: the test launcher.Supported values are test_runner.py, dist_test.sh and custom scripts' name.
    # * `dist_ut_port`: the starting port used in a distributed unit test
    # * `run_serial`: whether in serial mode. the value can be 1 or 0. Default(empty) is 0
    # * `ENVS`: required environments. multiple envirenmonts are splited by ";".
    # * `conditions`: extra required conditions for some tests. the value is a boolean expression in cmake programmer.
    name, os_, archs, timeout, run_type, launcher, dist_ut_port, run_serial, envs, conditions = line.strip(
    ).split(",")

    if name == "name":
        return ""

    envs = process_envs(envs)
    conditions = process_conditions(conditions)
    archs = proccess_archs(archs)
    os_ = process_os(os_)
    run_serial = process_run_serial(run_serial)

    cmd = ""

    if launcher[-3:] == ".sh":
        cmd += f'''if({archs} AND {os_} {conditions})
    bash_test_modules(
    {name}
    START_BASH
    {launcher}
    LABELS
    "RUN_TYPE={run_type}"
    ENVS
    "PADDLE_DIST_UT_PORT={dist_ut_port};{envs}")
    set_tests_properties({name} PROPERTIES  TIMEOUT "{timeout}" RUN_SERIAL {run_serial})
endif()
'''
    else:
        cmd += f'''if({archs} AND {os_} {conditions})
    py_test_modules(
    {name}
    MODULES
    {name}
    ENVS
    "PADDLE_DIST_UT_PORT={dist_ut_port};{envs}")
    set_tests_properties({name} PROPERTIES  TIMEOUT "{timeout}" RUN_SERIAL {run_serial})
endif()
'''
    return cmd


def gen_cmakelists(current_work_dir):
    print("procfessing dir:", current_work_dir)
    if current_work_dir == "":
        current_work_dir = "."
    cmds = """# This file is generated by ${PADDLE_ROOT}/tools/gen_ut_cmakelists.py.
# Please don't modify this file manually.
# If you need to change unittests in this file, please modify testslist.csv in the current directory 
# and then run the command `python3 ${PADDLE_ROOT}/tools/gen_ut_cmakelists.py -f ${CURRENT_DIRECTORY}/testslist.csv`
set(LOCAL_ALL_ARCH ON)
set(LOCAL_ALL_PLAT ON)\n"""
    with open(f"{current_work_dir}/testslist.csv") as csv_file:
        for line in csv_file:
            cmds += parse_line(line)
    print(cmds, end="")
    with open(f"{current_work_dir}/CMakeLists.txt", "w") as cmake_file:
        print(cmds, end="", file=cmake_file)


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        "-f",
        type=str,
        required=False,
        default=[],
        nargs="+",
        help=
        "Input a list of files named testslist.csv and output files named CmakeLists.txt in the same directories as the csv files respectly"
    )
    parser.add_argument(
        "--dirpaths",
        "-d",
        type=str,
        required=False,
        default=[],
        nargs="+",
        help=
        "Input a list of dir paths including files named testslist.csv and output CmakeLists.txt in these directories respectly"
    )
    args = parser.parse_args()

    assert not (len(args.files) == 0 and len(args.dirpaths)
                == 0), "You must provide at leate one file or dirpath"
    current_work_dirs = []
    if len(args.files) >= 1:
        for p in args.files:
            assert os.path.basename(
                p) == "testslist.csv", "you must input file named testslist.csv"
        current_work_dirs = current_work_dirs + [
            os.path.dirname(file) for file in args.files
        ]
    if len(args.dirpaths) >= 1:
        current_work_dirs = current_work_dirs + [d for d in args.dirpaths]

    for c in current_work_dirs:
        gen_cmakelists(c)
