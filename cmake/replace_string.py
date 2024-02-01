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

import sys


def main():
    src = sys.argv[1]

    with open(src, 'r') as file:
        content = file.read()

    new_content = content.replace('paddle.fluid', 'paddle.base')

    with open(src, 'w') as file:
        file.write(new_content)


if __name__ == "__main__":
    main()
