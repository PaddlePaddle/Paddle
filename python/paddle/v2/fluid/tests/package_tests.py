#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys, os
import re, io
import inspect
import tokenize
'''
NOTE(dzhwinter):
inspect.getmembers(module) and inspect.isroutine, inspect.isfunction is
useful. Consider paddle.core contains a lot symbols, it really cost a lot of time to inspect the source file. So I treat scripts as plain text, yeah, it's ugly.
'''

TEST_SINGLE_OPS = "test_\w+_op.py$"
INDENT = 2
INSIDE_MAIN = False


def _pipeline(stream, functors):
    global INDENT, INSIDE_MAIN
    for func in functors:
        stream = func(stream)
    return stream


def _find_indent(stream):
    global INDENT, INSIDE_MAIN
    snippets = [x for x in stream.split("\n") if len(x.strip()) != 0]
    for line in snippets:
        if line.startswith("class"):
            next_line = next(snippets)
            INDENT = len(next_line) - len(next_line.lstrip())
    return stream


def _rename_dup(stream):
    '''
    renmae duplicated global variable, global function
    '''
    global_names = {}

    def tokenize(func):
        assert (func.startswith("def"))
        p = 0  # current position
        left = 0
        right = len(func) - 1
        while p < len(func):
            if p + 1 < len(func):
                next_char = func[p + 1]
                if func[p] == ' ' and next_char.isalpha():
                    left = p + 1
            if func[p] == '(':
                right = p
        return func[left:right - left + 1]

    # python support declar function after using position,
    # so it must be two rounds
    for line in stream.split("\n"):
        if not line.startswith(" ") and not line.startswith("class"):
            token = tokenize(line)
            global_names.add(token)

    ans = ""
    for line in stream.split("\n"):
        for token in global_names:
            ans += line.replace("__" + token) + "\n"
    return ans


def _remove_main(stream):
    global INDENT, INSIDE_MAIN
    snippets = stream.split("\n")
    ans = ""
    for line in snippets:
        if "__main__" in line:
            INSIDE_MAIN = True
            continue
        if "unittest.main()" in line:
            continue
        if INSIDE_MAIN:
            ans += " " * INDENT + line + "\n"
    return ans


def main():
    files = os.listdir()
    print(files)
    sout = ""
    trigger_files = []
    for f in files:
        if re.match(TEST_SINGLE_OPS, f):
            print(f)
            trigger_files.append(f)
            original_contents = io.open(f, encoding="utf-8").read()
            # sout += _pipeline(original_contents,
            #                 [_find_indent, _rename_dup, _remove_main])

    print(" ".join(trigger_files) + " merged into single test.")
    sys.stdout.write(sout)


if __name__ == '__main__':
    main()
