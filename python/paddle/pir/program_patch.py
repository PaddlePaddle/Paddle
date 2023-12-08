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

from . import Program

_already_patch_program = False

global_prog_seed = 0


def monkey_patch_program():
    def global_seed(self, seed=0):
        global global_prog_seed
        global_prog_seed = seed
        self._seed = global_prog_seed

    global _already_patch_program
    if not _already_patch_program:
        Program.global_seed = global_seed
        global global_prog_seed
        Program._seed = global_prog_seed

        _already_patch_program = True
