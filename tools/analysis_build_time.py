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

import os

root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))


def strToSecond(strTime):
    minute = int(strTime.split(':')[0])
    second = int(strTime.split(':')[1].split('.')[0]) + 1
    return minute * 60 + second


def getUsefulBuildTimeFile(filename):
    os.system(
        f"grep -Po -- '-o .*' {filename} | grep ' elapsed' | grep -P -v '0:00.* elapse' > {root_path}/tools/analysis_build_time"
    )
    os.system(
        f"grep -v  -- '-o .*' {filename} |grep ' elapse' |  grep -P -v '0:00.* elapse' >> {root_path}/tools/analysis_build_time"
    )


def analysisBuildTime():
    filename = f'{root_path}/build/build-time'
    getUsefulBuildTimeFile(filename)
    os.system(f'rm -rf {root_path}/tools/tempbuildTime.txt')
    with open(f'{root_path}/tools/analysis_build_time', 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                line = line.strip()
                if '-o ' in line:
                    buildFile = line.split(', ')[0].split(' ')[1]
                    buildTime = line.split(', ')[1].split('elapsed')[0].strip()
                    secondTime = strToSecond(buildTime)
                    os.system(
                        f"echo {buildFile}, {secondTime} >> {root_path}/tools/tempbuildTime.txt"
                    )
                else:
                    buildTime = line.split(', ')[1].split('elapsed')[0].strip()
                    secondTime = strToSecond(buildTime)
                    if secondTime > 30:
                        os.system(
                            f"echo {line}, {secondTime} >> {root_path}/tools/tempbuildTime.txt"
                        )
            except ValueError:
                print(line)
    os.system(
        f'sort -n -k 2 -r {root_path}/tools/tempbuildTime.txt > {root_path}/tools/buildTime.txt'
    )


analysisBuildTime()
