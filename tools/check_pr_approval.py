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

from __future__ import print_function
import sys
import json


def check_approval(count):
    json_buff = ""
    for line in sys.stdin:
        json_buff = "".join([json_buff, line])
    json_resp = json.loads(json_buff)
    approves = 0
    for review in json_resp:
        if review["state"] == "APPROVED":
            approves += 1

    if approves >= count:
        print("TRUE")
    else:
        print("FALSE")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        check_approval(int(sys.argv[1]))
    else:
        print("Usage: python check_pr_approval.py [count]")
