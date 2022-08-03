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


def parse_line(line):
    name, os_, arch, timeout, run_type, launcher, dist_ut_port, envs, conditions = line.strip(
    ).split(",")
    if len(conditions.strip()) == 0:
        conditions = ""
    else:
        conditions = f" AND ({conditions})"
    archs = ""
    if len(arch.strip()) > 0:
        for a in arch.split(";"):
            archs += "WITH_" + a.upper() + " OR "
        arch = "(" + archs[:-4] + ")"
    else:
        arch = "LOCAL_ALL_ARCH"

    cmd = ""
    if name == "name":
        return cmd
    if len(os_.strip()) > 0:
        os_ = os_.replace(";", " or ")
        os_ = os_.upper()
        os_ = os_.replace("LINUX", "(NOT APPLE AND NOT WIN32)")
        os_ = "(" + os_ + ")"
    else:
        os_ = "LOCAL_ALL_PLAT"
    a = arch
    if launcher[-3:] == ".sh":
        cmd += f'''if({a} AND {os_} {conditions})
    bash_test_modules(
    {name}
    START_BASH
    {launcher}
    LABELS
    "RUN_TYPE={run_type}"
    ENVS
    "PADDLE_DIST_UT_PORT={dist_ut_port};{envs}")
    set_tests_properties({name} PROPERTIES  TIMEOUT "{timeout}")
endif()
'''
    else:
        cmd += f'''if({a} AND {os_} {conditions})
    py_test_modules(
    {name}
    MODULES
    {name}
    ENVS
    "PADDLE_DIST_UT_PORT={dist_ut_port};{envs}")
    set_tests_properties({name} PROPERTIES  TIMEOUT "{timeout}")
endif()
'''
    return cmd


def gen_cmakelists(current_work_dir):
    print("procfessing dir:", current_work_dir)
    if current_work_dir == "":
        current_work_dir = "."
    cmds = "set(LOCAL_ALL_ARCH ON)\nset(LOCAL_ALL_PLAT ON)\n"
    for line in open(f"{current_work_dir}/testslist.csv"):
        cmds += parse_line(line)
    print(cmds, end="")
    print(cmds, end="", file=open(f"{current_work_dir}/CMakeLists.txt", "w"))


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
        "input a files named testslist.csv and output a CmakeLists.txt in the same directory"
    )
    parser.add_argument(
        "--dirpaths",
        "-d",
        type=str,
        required=False,
        default=[],
        nargs="+",
        help=
        "input a dir path that including a file named testslist.csv and output a CmakeLists.txt in this directories"
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
        current_work_dir = current_work_dir + [d for d in args.dirpaths]

    for c in current_work_dirs:
        gen_cmakelists(c)
