# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import re


def escape(input):
    o = input.replace("\n", "")
    o = o.replace("\r", "")
    return o


def main():
    usage = """Usage:
1. Download the Paddle_PR_CI_*.log from TeamCity
2. run: python check_ctest_hung.py Paddle_PR_CI_*.log
3. If there is hung ctest, the result likes:
Diff:  set(['test_parallel_executor_crf'])
    """
    if len(sys.argv) < 2:
        print(usage)
        exit(0)

    logfile = sys.argv[1]
    started = set()
    passed = set()
    with open(logfile, "r") as fn:
        for l in fn.readlines():
            if l.find("Test ") != -1 and \
                l.find("Passed") != -1:
                m = re.search(r"Test\s+#[0-9]*\:\s([a-z0-9_]+)", escape(l))
                passed.add(m.group(1))
            if l.find("Start ") != -1:
                start_parts = escape(l).split(" ")
                m = re.search(r"Start\s+[0-9]+\:\s([a-z0-9_]+)", escape(l))
                started.add(m.group(1))
    print("Diff: ", started - passed)


if __name__ == "__main__":
    main()
