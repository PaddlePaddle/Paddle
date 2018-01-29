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
import keyword
import random
import string
'''
NOTE(dzhwinter):
inspect.getmembers(module) and inspect.isroutine, inspect.isfunction is
useful. Consider paddle.core contains a lot symbols,
inspect cost a lot of time to inspect the source file.
So I treat scripts as plain text.
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
    INDENT = 2
    snippets = [x for x in stream.split("\n") if len(x.strip()) != 0]
    snippets = iter(snippets)
    for line in snippets:
        if line.startswith("class"):
            next_line = next(snippets)
            INDENT = len(next_line) - len(next_line.lstrip())
    return stream


VALID_VARIABLE = "^[a-zA-Z_][a-zA-Z0-9_]*"


def _rename_dup(stream):
    '''
    renmae duplicated global variable, global function
    '''
    global_names = set()

    def tokenize(func):
        first_token = func.split()[0]
        # parse global function
        if keyword.iskeyword(first_token):
            if first_token != "def":
                return None
        # parse global variable
        if not re.match(VALID_VARIABLE, first_token):
            return None

        p = 0  # current position
        left = 0
        right = len(func) - 1
        while p < len(func):
            if p + 1 < len(func):
                next_char = func[p + 1]
                if func[p] == ' ' and next_char.isalpha() and left == 0:
                    left = p + 1
            if func[p] == '(' or func[p] == ' ' and right == len(func) - 1:
                right = p
            p += 1
        return func[left:right]

    # python support declar function after using position,
    # so it must be two rounds
    for line in stream.split("\n"):
        if line.startswith("def"):
            token = tokenize(line)
            if token != None and token != '':
                global_names.add(token)

    ans = ""
    random_prefix = ''.join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    for line in stream.split("\n"):
        line_ans = line
        if len(global_names) != 0:
            for token in global_names:
                line_ans = line_ans.replace(token,
                                            "_" + random_prefix + "_" + token)
        ans += line_ans + "\n"
    return ans


def _remove_main(stream):
    global INDENT, INSIDE_MAIN
    INSIDE_MAIN = False
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
        else:
            ans += line + "\n"
    return ans


def _append_main(stream):
    ans = '''
if __name__ == "__main__":
    unittest.main()
'''
    return stream + ans


def main():
    files = os.listdir()
    sout = ""
    trigger_files = []
    for f in files:
        if re.match(TEST_SINGLE_OPS, f):
            trigger_files.append(f)
            original_contents = io.open(f, encoding="utf-8").read()
            sout += _pipeline(original_contents,
                              [_find_indent, _rename_dup, _remove_main])

    sout = _append_main(sout)
    sys.stdout.write(sout)


if __name__ == '__main__':
    main()
