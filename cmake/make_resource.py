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

import os
import re
import sys

res = sys.argv[1]
out = sys.argv[2]
var = re.sub(r'[ .-]', '_', os.path.basename(res))

open(out, "w").write(
    "const unsigned char "
    + var
    + "[] = {"
    + ",".join([f"0x{ord(c):02x}" for c in open(res).read()])
    + ",0};\n"
    + "const unsigned "
    + var
    + "_size = sizeof("
    + var
    + ");\n"
)
