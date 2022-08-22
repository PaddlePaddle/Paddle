# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os

if __name__ == "__main__":
    assert len(sys.argv) == 3
    pybind_dir = sys.argv[1]
    split_count = int(sys.argv[2])

    empty_files = [os.path.join(pybind_dir, "eager_legacy_op_function.cc")]
    empty_files.append(os.path.join(pybind_dir, "eager_op_function.cc"))

    for i in range(split_count):
        empty_files.append(
            os.path.join(pybind_dir, "op_function" + str(i + 1) + ".cc"))

    for path in empty_files:
        if not os.path.exists(path):
            open(path, 'a').close()
