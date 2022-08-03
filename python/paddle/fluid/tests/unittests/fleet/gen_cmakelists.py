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


def parse_line(line, tests):
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
    if (name not in tests):
        tests.append(name)
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


import os
if __name__ == "__main__":
    current_work_dir = os.path.dirname(__file__)
    if current_work_dir == "":
        current_work_dir = "."
    tests = []
    cmds = "set(LOCAL_ALL_ARCH ON)\nset(LOCAL_ALL_PLAT ON)\n"
    for line in open(f"{current_work_dir}/testslist.csv"):
        cmds += parse_line(line, tests)
    print(cmds, end="")
    print(cmds, end="", file=open(f"{current_work_dir}/CMakeLists.txt", "w"))
