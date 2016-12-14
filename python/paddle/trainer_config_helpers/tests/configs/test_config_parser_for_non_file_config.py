#!/usr/bin/env python
# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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
import getopt


def main(print_whole_config, globals, locals):
    '''
     this test will all test_config.py
  '''
    cmdstr = """from paddle.trainer.config_parser import parse_config\n"""
    importstr = ""
    functionstr = ""

    for line in sys.stdin:
        if re.match("^import", line) or re.match("^from.*import", line):
            importstr = importstr + line
        else:
            functionstr = functionstr + "  " + line

    cmdstr = cmdstr + importstr + """def configs():\n""" + functionstr
    #cmdstr = cmdstr + """def configs():\n""" + importstr + functionstr
    if print_whole_config:
        cmdstr = cmdstr + """print parse_config(configs, "")"""
    else:
        cmdstr = cmdstr + """print parse_config(configs, "").model_config"""

    exec (cmdstr, globals, locals)


if __name__ == '__main__':
    whole = False
    opts, args = getopt.getopt(sys.argv[1:], "", ["whole"])
    for op, value in opts:
        if op == "--whole":
            whole = True
    main(whole, globals(), locals())
