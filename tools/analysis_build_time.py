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
        "grep -Po -- '-o .*' %s | grep ' elapsed' | grep -P -v '0:00.* elapse' > %s/tools/analysis_build_time"
        % (filename, root_path))
    os.system(
        "grep -v  -- '-o .*' %s |grep ' elapse' |  grep -P -v '0:00.* elapse' >> %s/tools/analysis_build_time"
        % (filename, root_path))


def analysisBuildTime():
    filename = '%s/build/build-time' % root_path
    getUsefulBuildTimeFile(filename)
    os.system('rm -rf %s/tools/tempbuildTime.txt' % root_path)
    with open('%s/tools/analysis_build_time' % root_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                line = line.strip()
                if '-o ' in line:
                    buildFile = line.split(', ')[0].split(' ')[1]
                    buildTime = line.split(', ')[1].split('elapsed')[0].strip()
                    secondTime = strToSecond(buildTime)
                    os.system("echo %s, %s >> %s/tools/tempbuildTime.txt" %
                              (buildFile, secondTime, root_path))
                else:
                    buildTime = line.split(', ')[1].split('elapsed')[0].strip()
                    secondTime = strToSecond(buildTime)
                    if secondTime > 30:
                        os.system("echo %s, %s >> %s/tools/tempbuildTime.txt" %
                                  (line, secondTime, root_path))
            except ValueError:
                print(line)
    os.system(
        'sort -n -k 2 -r %s/tools/tempbuildTime.txt > %s/tools/buildTime.txt' %
        (root_path, root_path))


analysisBuildTime()
