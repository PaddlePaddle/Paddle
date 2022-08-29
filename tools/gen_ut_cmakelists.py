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
import os
import argparse

# port range (21200, 23000) is reserved for dist-ops


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
        conditions = []
    else:
        conditions = conditions.strip().split(";")
    return [c.strip() for c in conditions]


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
            assert a in ["GPU", "ROCM", "ASCEND", "ASCEND_CL", "XPU"], \
                f"""Supported arhc options are "GPU", "ROCM", "ASCEND" and "ASCEND_CL", "XPU", but the options is {a}"""
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
        return ""
    return rs


def file_with_extension(prefix, suffixes):
    """
    Desc:
        check whether test file exists. 
    """
    for ext in suffixes:
        if os.path.isfile(prefix + ext):
            return True
    return False


def process_name(name, curdir):
    """
    Desc:
        check whether name is with a legal format and check whther the test file exists.
    """
    name = name.strip()
    assert re.compile("^test_[0-9a-zA-Z_]+").search(name), \
        f"""If line is not the header of table, the test name must begin with "test_" """ \
        f"""and the following substring must include at least one char of "0-9", "a-z", "A-Z" or "_"."""
    filepath_prefix = os.path.join(curdir, name)
    suffix = [".py", ".sh"]
    assert file_with_extension(filepath_prefix, suffix), \
        f""" Please ensure the test file with the prefix '{filepath_prefix}' and one of the suffix {suffix} exists, because you specified a unittest named '{name}'"""

    return name


def process_run_type(run_type):
    rt = run_type.strip()
    assert re.compile("^(NIGHTLY|EXCLUSIVE|CINN|DIST|GPUPS|INFER|EXCLUSIVE:NIGHTLY|DIST:NIGHTLY)$").search(rt), \
        f""" run_type must be one of 'NIGHTLY', 'EXCLUSIVE', 'CINN', 'DIST', 'GPUPS', 'INFER', 'EXCLUSIVE:NIGHTLY' and 'DIST:NIGHTLY'""" \
        f"""but the run_type is {rt}"""
    return rt


DIST_UT_PORT = 21200
ASSIGNED_PORTS = dict()


def gset_port(test_name, port):
    '''
    Get and set a port for unit test named test_name. If the test has been already holding a port, return the port it holds.
    Else assign the input port as a new port to the test.
    '''
    if test_name not in ASSIGNED_PORTS:
        ASSIGNED_PORTS[test_name] = port
    global DIST_UT_PORT
    DIST_UT_PORT = max(DIST_UT_PORT, ASSIGNED_PORTS[test_name])
    return ASSIGNED_PORTS[test_name]


def process_dist_port_num(port_num):
    assert re.compile("^[0-9]+$").search(port_num) and int(port_num) > 0 or port_num.strip()=="", \
        f"""port_num must be foramt as a positive integer or empty, but this port_num is '{port_num}'"""
    port_num = port_num.strip()
    if len(port_num) == 0:
        return 0
    global DIST_UT_PORT
    port = DIST_UT_PORT
    assert port < 23000, "dist port is exhausted"
    DIST_UT_PORT += int(port_num)
    return port


def parse_line(line, curdir):
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

    name, os_, archs, timeout, run_type, launcher, num_port, run_serial, envs, conditions = line.strip(
    ).split(",")

    # name == "name" means the line being parsed is the header of the table
    # we should skip this line and return empty here.
    if name == "name":
        return ""
    name = process_name(name, curdir)

    envs = process_envs(envs)
    conditions = process_conditions(conditions)
    archs = proccess_archs(archs)
    os_ = process_os(os_)
    run_serial = process_run_serial(run_serial)
    run_type = process_run_type(run_type)

    cmd = ""

    for c in conditions:
        cmd += f"if ({c})\n"

    if launcher[-3:] == ".sh":
        dist_ut_port = process_dist_port_num(num_port)
        dist_ut_port = gset_port(name, dist_ut_port)
        cmd += f'''if({archs} AND {os_})
    bash_test_modules(
    {name}
    START_BASH
    {launcher}
    LABELS
    "RUN_TYPE={run_type}"
    ENVS
    "PADDLE_DIST_UT_PORT={dist_ut_port};{envs}")%s
endif()
'''
    else:
        cmd += f'''if({archs} AND {os_})
    py_test_modules(
    {name}
    MODULES
    {name}
    ENVS
    "{envs}")%s
endif()
'''
    time_out_str = f' TIMEOUT "{timeout}"' if len(timeout.strip()) > 0 else ''
    run_serial_str = f' RUN_SERIAL {run_serial}' if len(run_serial) > 0 else ''
    if len(time_out_str) > 0 or len(run_serial_str) > 0:
        set_properties = f'''
    set_tests_properties({name} PROPERTIES{time_out_str}{run_serial_str})'''
    else:
        set_properties = ""
    cmd = cmd % set_properties
    for _ in conditions:
        cmd += f"endif()\n"
    return cmd


PROCESSED_DIR = set()

LAST_TEST_NAME = ""
LAST_TEST_CMAKE_FILE = ""


def init_dist_ut_ports_from_cmakefile(cmake_file_name):
    '''
    Find all signed ut ports in cmake_file and update the ASSIGNED_PORTS
    and keep the DIST_UT_PORT max of all assigned ports 
    '''
    with open(cmake_file_name) as cmake_file:
        port_reg = re.compile("PADDLE_DIST_UT_PORT=[0-9]+")
        lines = cmake_file.readlines()
        for idx, line in enumerate(lines):
            matched = port_reg.search(line)
            if matched is not None:
                p = matched.span()
                port = int(line[p[0]:p[1]].split("=")[-1])
                for k in range(idx, 0, -1):
                    if lines[k].strip() == "START_BASH":
                        break
                name = lines[k - 1].strip()
                assert re.compile("^test_[0-9a-zA-Z_]+").search(name), \
                    f'''we found a test for initial the latest dist_port but the test name '{name}' seems to be wrong
                    at line {k-1}, in file {cmake_file_name}
                    '''
                gset_port(name, port)
                if ASSIGNED_PORTS[name] == DIST_UT_PORT:
                    global LAST_TEST_CMAKE_FILE, LAST_TEST_NAME
                    LAST_TEST_NAME = name
                    LAST_TEST_CMAKE_FILE = cmake_file_name


NO_CMAKE_DIR_WARNING = []


def parse_assigned_dist_ut_ports(current_work_dir, ignores, depth=0):
    '''
    get all assigned dist ports to keep port of unmodified test fixed.
    '''
    if current_work_dir in PROCESSED_DIR:
        return
    if depth == 0:
        ignores = [os.path.abspath(i) for i in ignores]
    PROCESSED_DIR.add(current_work_dir)
    contents = os.listdir(current_work_dir)
    cmake_file = os.path.join(current_work_dir, "CMakeLists.txt")
    csv = cmake_file.replace("CMakeLists.txt", 'testslist.csv')
    if os.path.isfile(csv) or os.path.isfile(cmake_file):
        if current_work_dir not in ignores:
            if os.path.isfile(cmake_file) and os.path.isfile(csv):
                init_dist_ut_ports_from_cmakefile(cmake_file)
            elif not os.path.isfile(cmake_file):
                NO_CMAKE_DIR_WARNING.append(current_work_dir)
        for c in contents:
            c_path = os.path.join(current_work_dir, c)
            if os.path.isdir(c_path):
                print("parsed ", c_path)
                parse_assigned_dist_ut_ports(c_path, ignores, depth + 1)
    if depth == 0:
        print("LAST_TEST_NAME:", LAST_TEST_NAME)
        if len(LAST_TEST_NAME) > 0 and len(LAST_TEST_CMAKE_FILE) > 0:
            with open(
                    LAST_TEST_CMAKE_FILE.replace("CMakeLists.txt",
                                                 "testslist.csv")) as csv_file:
                found = False
                for line in csv_file.readlines():
                    name, _, _, _, _, launcher, num_port, _, _, _ = line.strip(
                    ).split(",")
                    if name == LAST_TEST_NAME:
                        found = True
                        break
            assert found, f"no such test named '{LAST_TEST_NAME}' in file '{LAST_TEST_CMAKE_FILE}'"
            if launcher[-2:] == ".sh":
                process_dist_port_num(num_port)

        err_msg = f"""==================[No Old CMakeLists.txt Error]==================================
    Following directories has no CmakeLists.txt files:
"""
        for c in NO_CMAKE_DIR_WARNING:
            err_msg += "   " + c + "\n"
        err_msg += """
      This may cause the dist ports different with the old version.
      If the directories are newly created or there is no CMakeLists.txt before, or ignore this error, you
    must specify the directories using the args option --ignore-cmake-dirs/-i.
      If you want to keep the dist ports of old tests unchanged, please ensure the old
    verson CMakeLists.txt file existing before using the gen_ut_cmakelists tool to 
    generate new CmakeLists.txt files.
====================================================================================
"""
        assert len(NO_CMAKE_DIR_WARNING) == 0, err_msg


def gen_cmakelists(current_work_dir):
    print("procfessing dir:", current_work_dir)
    if current_work_dir == "":
        current_work_dir = "."

    contents = os.listdir(current_work_dir)
    contents.sort()
    sub_dirs = []
    for c in contents:
        c_path = os.path.join(current_work_dir, c)
        if c_path in PROCESSED_DIR:
            return
        if os.path.isdir(c_path):
            PROCESSED_DIR.add(c_path)
            if os.path.isfile(os.path.join(current_work_dir, c, "testslist.csv")) \
                or os.path.isfile(os.path.join(current_work_dir, c, "CMakeLists.txt")):
                gen_cmakelists(os.path.join(current_work_dir, c))
                sub_dirs.append(c)

    if not os.path.isfile(os.path.join(current_work_dir, "testslist.csv")):
        return
    cmds = """# This file is generated by ${PADDLE_ROOT}/tools/gen_ut_cmakelists.py.
# Please don't modify this file manually.
# If you need to change unittests in this file, please modify testslist.csv in the current directory 
# and then run the command `python3 ${PADDLE_ROOT}/tools/gen_ut_cmakelists.py -f ${CURRENT_DIRECTORY}/testslist.csv`
set(LOCAL_ALL_ARCH ON)
set(LOCAL_ALL_PLAT ON)\n"""
    with open(f"{current_work_dir}/testslist.csv") as csv_file:
        for i, line in enumerate(csv_file.readlines()):
            try:
                cmds += parse_line(line, current_work_dir)
            except Exception as e:
                print("===============PARSE LINE ERRORS OCCUR==========")
                print(e)
                print(f"[ERROR FILE]: {current_work_dir}/testslist.csv")
                print(f"[ERROR LINE {i+1}]: {line.strip()}")
                exit(1)

    for sub in sub_dirs:
        cmds += f"add_subdirectory({sub})\n"
    print(cmds, end="")
    with open(f"{current_work_dir}/CMakeLists.txt", "w") as cmake_file:
        print(cmds, end="", file=cmake_file)


if __name__ == "__main__":
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
    parser.add_argument(
        "--ignore-cmake-dirs",
        '-i',
        type=str,
        required=False,
        default=[],
        nargs='*',
        help=
        "To keep dist ports the same with old version cmake, old cmakelists.txt files are needed to parse dist_ports. If a directories are newly created and there is no cmakelists.txt file, the directory path must be specified by this option. The dirs are not recursive."
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
        # c = os.path.abspath(c)
        while True:
            print(c)
            dirpath = os.path.dirname(c)
            cmake = os.path.join(dirpath, "CMakeLists.txt")
            csv = os.path.join(dirpath, "testslist.csv.txt")
            if not (os.path.isfile(cmake) or os.path.isfile(csv)):
                break
            c = os.path.abspath(dirpath)
        print("=============>>>>>>", c)
        parse_assigned_dist_ut_ports(c, ignores=args.ignore_cmake_dirs, depth=0)

    PROCESSED_DIR.clear()
    for c in current_work_dirs:
        c = os.path.abspath(c)
        gen_cmakelists(c)
