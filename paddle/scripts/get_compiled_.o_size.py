# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import subprocess


class handlePaddle:
    def __init__(self) -> None:
        cmd = "find -name '*.o' -not -path './third_party/*' | xargs du -sch > log"
        subprocess.run(cmd, shell=True)

    def getFile(self, filename='log', mode='r'):
        with open(filename, mode) as file:
            ctx = file.read()
        ctx = ctx.split('\n')
        vec = []
        sum = 0
        for item in ctx:
            if item.find('total') >= 0:
                item = item.split('\t')[0]
                if item.find('G') >= 0:
                    sum += float(item[:-1]) * 1024
                else:
                    sum += float(item[:-1])
                # sum += int(item)

        print(f"Total size is {sum} M (without third_party)")


if __name__ == '__main__':
    handlepaddle = handlePaddle()
    handlepaddle.getFile()
