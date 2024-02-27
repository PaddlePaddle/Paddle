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


from . import Block

_already_patch_block = False


def monkey_patch_block():
    def all_parameters(self):
        return self.params

    block_attrs = {
        "all_parameters": all_parameters,
        "params": [],
    }

    global _already_patch_block
    if not _already_patch_block:
        for attr, value in block_attrs.items():
            setattr(Block, attr, value)

        _already_patch_block = True
