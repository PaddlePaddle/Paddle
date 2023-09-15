# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,tes
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
        # opresult -> list(list(opresult))
        self.value_to_valuegrad = collections.defaultdict(list)
        self.value_to_sumvaluegrad = collections.defaultdict(list)
        # operation -> list(operation)
        self.op_to_opgrad = collections.defaultdict(list)

        # opresult -> list(opresult)
        self.valuegrad_to_value = collections.defaultdict(list)
        self.sumvaluegrad_to_value = collections.defaultdict(list)
        # operation -> list(operation)
        self.opgrad_to_op = collections.defaultdict(list)

    def turn_map(self) -> None:
        self.valuegrad_to_value = collections.defaultdict(list)
        self.sumvaluegrad_to_value = collections.defaultdict(list)
        self.opgrad_to_op = collections.defaultdict(list)

        for k, v in self.value_to_valuegrad.items():
            if v != []:
                for value in v[0]:
                    self.valuegrad_to_value[value] = [k]
        for k, v in self.value_to_sumvaluegrad.items():
            if v != []:
                for value in v[0]:
                    self.sumvaluegrad_to_value[value] = [k]
        for k, v in self.op_to_opgrad.items():
            if v != []:
                self.opgrad_to_op[v[0]] = [k]
