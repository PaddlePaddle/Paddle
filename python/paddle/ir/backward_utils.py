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

import collections


class State:
    """
    record relationship of forward op/value and backward op/value
    one state must be bining with a program

    """

    def __init__(self, program):
        self.program = program
        self.value_to_valuegrad = collections.defaultdict(list)
        self.value_to_sumvaluegrad = collections.defaultdict(list)
        self.op_to_opgrad = collections.defaultdict(list)
        self.valuegrad_to_value = collections.defaultdict(list)
        self.sumvaluegrad_to_value = collections.defaultdict(list)
        self.opgrad_to_op = collections.defaultdict(list)

    def turn_map(self) -> None:
        for k, v in self.value_to_valuegrad.items():
            for value in v:
                self.valuegrad_to_value[value] = k
        for k, v in self.value_to_sumvaluegrad.items():
            for value in v:
                self.sumvaluegrad_to_value[value] = k
        for k, v in self.op_to_opgrad.items():
            for value in v:
                self.opgrad_to_op[value] = k
