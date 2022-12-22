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

import re
from type_mapping import input_types_map, attr_types_map, output_type_map


# tests for typename
def is_input(s):
    return s in input_types_map


def is_attr(s):
    return s in attr_types_map


def is_output(s):
    return s in output_type_map


def is_vec(s):
    return s.endswith("[]")


def is_scalar(s):
    return re.match(r"Scalar(\(\w+\))*", s) is not None


def is_initializer_list(s):
    return s == "{}"


def is_base_op(op):
    return "kernel" in op and "infer_meta" in op


def supports_selected_rows_kernel(op):
    return is_base_op(op) and len(op["kernel"]["func"]) == 2


def supports_inplace(op):
    return op['inplace'] is not None


def supports_no_need_buffer(op):
    for input in op["inputs"]:
        if input["no_need_buffer"]:
            return True
    return False
