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


class NameGenerator:
    def __init__(self):
        self.key2name = {}
        self.prefix2counter = {}

    def Generate(self, key, prefix):
        if key in self.key2name:
            return self.key2name[key]
        if prefix not in self.prefix2counter:
            self.prefix2counter[prefix] = -1
        self.prefix2counter[prefix] += 1
        name = f"{prefix}_{self.prefix2counter[prefix]}"
        self.key2name[key] = name
        return name
