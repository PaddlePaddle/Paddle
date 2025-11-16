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


import json
import sys


def main():
    comment_body = sys.argv[1]
    mapping_file = sys.argv[2]

    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    command = comment_body.replace('/re-run', '').strip()

    # Exact match first
    if command in mapping:
        print(mapping[command])
        return

    # Partial match for compound commands (e.g., "inference build")
    # Sort by key length descending to match "inference build" before "inference"
    for keyword in sorted(mapping.keys(), key=len, reverse=True):
        # Simple keyword check
        if all(word in command.split() for word in keyword.split()):
            print(mapping[keyword])
            return

    print()


if __name__ == "__main__":
    main()
