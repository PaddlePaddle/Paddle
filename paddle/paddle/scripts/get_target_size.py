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
import sys


class HandleTarget:
    def __init__(self) -> None:
        argv = sys.argv
        len_ = len(argv)
        fromPath = './' if len_ <= 1 else argv[1]
        cmd = f"find {fromPath} -name '*.o' -not -path './third_party/*' | xargs du -sch > log"
        subprocess.run(cmd, shell=True)

    def calcuSize(self, item):
        size = float(item[:-1])
        res = size * 1024 if item.find('G') >= 0 else size
        return size / 1024 if item.find('K') >= 0 else res

    def getSize(self):
        ctx = self.getDatas()
        sum = 0
        for item in ctx:
            if item.find('total') >= 0:
                item = item.split('\t')[0]
                sum += self.calcuSize(item)

        print(f"Total size is {sum} M (without third_party)")

    def getDatas(self):
        with open('log', 'r') as file:
            ctx = file.read()
        ctx = ctx.split('\n')
        return ctx


if __name__ == '__main__':
    handler = HandleTarget()
    handler.getSize()
