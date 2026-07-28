# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import ap


class NumberOfInputsTrait0:
    def number_of_inputs(self):
        return 0


class NumberOfInputsTrait1:
    def number_of_inputs(self):
        return 1


class NumberOfInputsTrait2:
    def number_of_inputs(self):
        return 2


class NumberOfInputsTrait3:
    def number_of_inputs(self):
        return 3


class NumberOfInputsTrait4:
    def number_of_inputs(self):
        return 4


class NumberOfInputsTrait5:
    def number_of_inputs(self):
        return 5


class NumberOfInputsTrait6:
    def number_of_inputs(self):
        return 6


class NumberOfInputsTrait7:
    def number_of_inputs(self):
        return 7


class NumberOfInputsTrait8:
    def number_of_inputs(self):
        return 8


class NumberOfInputsTrait9:
    def number_of_inputs(self):
        return 9


class NumberOfInputsTrait10:
    def number_of_inputs(self):
        return 10


class NumberOfInputsTrait11:
    def number_of_inputs(self):
        return 11


class NumberOfInputsTrait12:
    def number_of_inputs(self):
        return 12


class NumberOfInputsTrait13:
    def number_of_inputs(self):
        return 13


class NumberOfInputsTrait14:
    def number_of_inputs(self):
        return 14


class NumberOfInputsTrait15:
    def number_of_inputs(self):
        return 15


class NumberOfInputsTrait16:
    def number_of_inputs(self):
        return 16


class NumberOfInputsTrait17:
    def number_of_inputs(self):
        return 17


class NumberOfOutputsTrait0:
    def number_of_outputs(self):
        return 0


class NumberOfOutputsTrait1:
    def number_of_outputs(self):
        return 1


class NumberOfOutputsTrait2:
    def number_of_outputs(self):
        return 2


class NumberOfOutputsTrait3:
    def number_of_outputs(self):
        return 3


class NumberOfOutputsTrait4:
    def number_of_outputs(self):
        return 4


class NumberOfOutputsTrait5:
    def number_of_outputs(self):
        return 5


class NumberOfOutputsTrait6:
    def number_of_outputs(self):
        return 6


class NumberOfOutputsTrait7:
    def number_of_outputs(self):
        return 7


class NumberOfOutputsTrait8:
    def number_of_outputs(self):
        return 8


class NumberOfOutputsTrait9:
    def number_of_outputs(self):
        return 9


class NumberOfOutputsTrait10:
    def number_of_outputs(self):
        return 10


class NumberOfOutputsTrait11:
    def number_of_outputs(self):
        return 11


class NumberOfOutputsTrait12:
    def number_of_outputs(self):
        return 12


class NumberOfOutputsTrait13:
    def number_of_outputs(self):
        return 13


class NumberOfOutputsTrait14:
    def number_of_outputs(self):
        return 14


class NumberOfOutputsTrait15:
    def number_of_outputs(self):
        return 15


class NumberOfOutputsTrait16:
    def number_of_outputs(self):
        return 16


class NumberOfOutputsTrait17:
    def number_of_outputs(self):
        return 17


class NumberOfOutputsTrait18:
    def number_of_outputs(self):
        return 18


class NumberOfOutputsTrait19:
    def number_of_outputs(self):
        return 19


class NumberOfOutputsTrait20:
    def number_of_outputs(self):
        return 20


class NumberOfOutputsTrait21:
    def number_of_outputs(self):
        return 21


class NumberOfOutputsTrait22:
    def number_of_outputs(self):
        return 22


def get_mixin_class(
    base_class, class_name_prefix, number_of_inputs, number_of_outputs
):
    num_inputs_to_input_trait_class = [
        None,
        NumberOfInputsTrait1,
        NumberOfInputsTrait2,
        NumberOfInputsTrait3,
        NumberOfInputsTrait4,
        NumberOfInputsTrait5,
        NumberOfInputsTrait6,
        NumberOfInputsTrait7,
        NumberOfInputsTrait8,
        NumberOfInputsTrait9,
        NumberOfInputsTrait10,
        NumberOfInputsTrait11,
        NumberOfInputsTrait12,
        NumberOfInputsTrait13,
        NumberOfInputsTrait14,
        NumberOfInputsTrait15,
        NumberOfInputsTrait16,
        NumberOfInputsTrait17,
    ]
    num_outputs_to_output_trait_class = [
        None,
        NumberOfOutputsTrait1,
        NumberOfOutputsTrait2,
        NumberOfOutputsTrait3,
        NumberOfOutputsTrait4,
        NumberOfOutputsTrait5,
        NumberOfOutputsTrait6,
        NumberOfOutputsTrait7,
        NumberOfOutputsTrait8,
        NumberOfOutputsTrait9,
        NumberOfOutputsTrait10,
        NumberOfOutputsTrait11,
        NumberOfOutputsTrait12,
        NumberOfOutputsTrait13,
        NumberOfOutputsTrait14,
        NumberOfOutputsTrait15,
        NumberOfOutputsTrait16,
        NumberOfOutputsTrait17,
        NumberOfOutputsTrait18,
        NumberOfOutputsTrait19,
        NumberOfOutputsTrait20,
        NumberOfOutputsTrait21,
        NumberOfOutputsTrait22,
    ]
    return type(
        f"{class_name_prefix}{number_of_inputs}_{number_of_outputs}",
        [
            base_class,
            num_inputs_to_input_trait_class[number_of_inputs],
            num_outputs_to_output_trait_class[number_of_outputs],
        ],
        ap.SerializableAttrMap(),
    )
