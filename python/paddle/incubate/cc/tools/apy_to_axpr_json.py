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
import glob
import os
import sys

from paddle.incubate.cc.ap.apy_to_axpr_json import PyToAnfParser


def PyToAxpr(filepath):
    if os.path.isdir(filepath):
        for file in glob.glob(f"{filepath}/*"):
            PyToAxpr(file)
    elif filepath.endswith(".py"):
        print(f"apy_to_axpr_json {filepath}")
        tree = ast.parse(open(filepath).read())
        parser = PyToAnfParser()
        parser(tree).ConvertToAnfExpr().DumpToFileAsJson(f"{filepath}.json")
    else:
        # Do nothing
        pass


if __name__ == "__main__":

    for filepath in sys.argv[1:]:
        PyToAxpr(filepath)
