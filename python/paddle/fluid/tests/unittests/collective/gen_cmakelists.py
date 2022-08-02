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
    name, os, arch, timeout, run_type, launcher, dist_ut_port, envs = line.strip(
    ).split(",")
    cmd = ""
    if name == "name":
        return cmd
    if (name not in tests):
        tests.append(name)
        for o in os.split(";"):
            o = o.upper()
            for a in arch.split(";"):
                a = a.upper()
                if (launcher[-3:] == ".sh"):
                    cmd = f'''
if(WITH_{a} AND {o})
    bash_test_modules(
    {name}
    START_BASH
    {launcher}
    LABELS
    "RUN_TYPE={run_type}"
    ENVS
    "PADDLE_DIST_UT_PORT={dist_ut_port};{envs};TIMEOUT={timeout}")
endif()
        '''
                else:
                    cmd = f'''
if(WITH_{a} AND {o})
    py_test_modules(
    {name}
    MODULES
    {name}
    ENVS
    "PADDLE_DIST_UT_PORT={dist_ut_port};{envs};TIMEOUT={timeout}")
endif()
    '''
    return cmd


tests = []
cmds = ""
for line in open("testslist.csv"):
    cmds += parse_line(line, tests)
print(cmds)
