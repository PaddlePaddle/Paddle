# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import ast
import fnmatch
import glob
import os
import sys
from pathlib import Path

from paddle.incubate.cc.ap.apy_to_axpr_json import PyToAnfParser

FLAGS_FIRST_CYCLE = True
APY_ROOT = "apy"


def CollectParentIgnoreFile(start_path):
    apy_ignores = []
    path = Path(start_path).parent
    while True:
        ignore_path = os.path.join(path, '.apy_ignore')
        if os.path.isfile(ignore_path):
            apy_ignores.append(ignore_path)
        parent_path = os.path.dirname(path)
        if parent_path.split(os.sep)[-1] == APY_ROOT or parent_path == path:
            break
        path = parent_path
    for root, dirs, files in os.walk(start_path):
        if '.apy_ignore' in files:
            apy_ignores.append(os.path.join(root, '.apy_ignore'))
    return apy_ignores


def ReadIgnoreRules(ignore_paths):
    rules = []
    for ignore_path in ignore_paths:
        base_dir = os.path.dirname(ignore_path)
        with open(ignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                is_negation = line.startswith('!')
                if is_negation:
                    pattern = line[1:]
                else:
                    pattern = line
                if pattern.startswith(os.sep):
                    pattern = pattern[1:]
                rules.append((pattern, is_negation, base_dir))
    return rules


def IsAllowed(file_path, ignore_rules):
    if os.path.isfile(file_path) and not file_path.endswith(".py"):
        return False
    file_path = os.path.normpath(file_path)
    result = True
    for pattern, is_negation, base_dir in ignore_rules:
        full_pattern = os.path.join(base_dir, pattern).rstrip(os.sep)
        if os.path.isdir(full_pattern):
            full_pattern += '*'
        match = fnmatch.fnmatch(file_path, full_pattern)
        if match:
            if is_negation:
                result = True
            else:
                result = False
    return result


class PyToAxpr:
    def __init__(self, file_path, ignore_paths=None):
        ignore_files = CollectParentIgnoreFile(file_path)
        self.ignore_rules = ReadIgnoreRules(ignore_files)
        if ignore_paths:
            self.ignore_rules += [(name, False, "") for name in ignore_paths]

    def __call__(self, file_path):
        if not IsAllowed(file_path, self.ignore_rules):
            pass
        elif os.path.isdir(file_path):
            for file in glob.glob(f"{file_path}{os.sep}*"):
                self.__call__(file)
        else:
            print(f"apy_to_axpr_json {file_path}")
            tree = ast.parse(open(file_path).read())
            parser = PyToAnfParser()
            parser(tree).ConvertToAnfExpr().DumpToFileAsJson(
                f"{file_path}.json"
            )


if __name__ == "__main__":
    for file_path in sys.argv[1:]:
        PyToAxpr(file_path)(file_path)
