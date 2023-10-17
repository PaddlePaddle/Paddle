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

import re
import sys

runtime_error_msg = sys.stdin.read()

pattern = r'File "?(.*?)"?, line (\d+),.*\n(.*?)\n(.*?)$'
for match in re.finditer(pattern, runtime_error_msg, re.MULTILINE):
    file = match.group(1)
    if file.startswith("./"):
        file = f"tests/{file[2:]}"
        line = match.group(2)
        error_info = match.group(4)
        if "AssertionError" not in error_info:
            # error_info = match.group(3) + '\n' + match.group(4)
            output = f"::error file={file},line={line}::Error"
            print(output)
