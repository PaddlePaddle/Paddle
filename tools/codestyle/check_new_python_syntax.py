# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import argparse
import ast
import os
from typing import Sequence


def check_file(filename: str) -> int:
    try:
        if not os.path.exists(filename):
            return 0
        with open(filename, 'rb') as f:
            ast_obj = ast.parse(f.read(), filename=filename)
    except SyntaxError:
        print(f'{filename} - ast Parsing failed')
        return 1

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*', help='Filenames to run')
    args = parser.parse_args(argv)

    retv = 0
    for filename in args.filenames:
        print(f"check: {filename}")
        retv |= check_file(filename)
    return retv


if __name__ == '__main__':
    raise SystemExit(main())
