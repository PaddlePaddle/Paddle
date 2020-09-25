# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
""" Check whether ut is disabled. """

import os
import sys


def check_ut():
    """ Get disabled unit tests. """
    disable_ut_file = 'disable_ut'
    cmd = 'wget -q --no-check-certificate https://sys-p0.bj.bcebos.com/prec/{}'.format(
        disable_ut_file)
    os.system(cmd)
    with open(disable_ut_file) as utfile:
        for u in utfile:
            if u.rstrip('\r\n') == sys.argv[1]:
                exit(0)
    exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit(1)
    try:
        check_ut()
    except Exception as e:
        print(e)
        exit(1)
