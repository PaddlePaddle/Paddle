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

from __future__ import print_function


def add_listen_and_serve_pass(program, origin_program):
    pass


def add_dist_info_pass(program, origin_program):
    program.random_seed = origin_program.random_seed
    program._copy_dist_param_info_from(origin_program)
    return program


def add_optimize_sub_block_pass(program):
    pass
