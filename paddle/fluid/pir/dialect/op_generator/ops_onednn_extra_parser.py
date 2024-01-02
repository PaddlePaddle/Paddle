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

import re
from typing import Any, Dict, List, Tuple


def parse_plain_list(s: str, sep=",") -> List[str]:
    if sep == ",":
        patten = re.compile(r',(?![^{]*\})')  # support "int[] a={1,2}"
        items = re.split(patten, s.strip())
        items = [x.strip() for x in items]
        return items
    else:
        return [item.strip() for item in s.strip().split(sep)]


def parse_arg(op_name: str, s: str) -> Dict[str, str]:
    """parse an argument in following formats:
    1. typename name
    2. typename name = default_value
    """
    typename, rest = (item.strip() for item in s.split(" ", 1))
    assert (
        len(typename) > 0
    ), f"The arg typename should not be empty. Please check the args of {op_name} in yaml."

    assert (
        rest.count("=") <= 1
    ), f"There is more than 1 = in an arg in {op_name}"
    if rest.count("=") == 1:
        name, default_value = (item.strip() for item in rest.split("=", 1))
        assert (
            len(name) > 0
        ), f"The arg name should not be empty. Please check the args of {op_name} in yaml."
        assert (
            len(default_value) > 0
        ), f"The default value should not be empty. Please check the args of {op_name} in yaml."
        return {
            "typename": typename,
            "name": name,
            "default_value": default_value,
        }
    else:
        name = rest.strip()
        assert (
            len(name) > 0
        ), f"The arg name should not be empty. Please check the args of {op_name} in yaml."
        return {"typename": typename, "name": name}


def parse_extra_args(op_name: str, arguments: str) -> List:
    if arguments is None:
        return []
    args_str = arguments.strip()
    args = parse_plain_list(args_str)

    attrs = []

    for arg in args:
        item = parse_arg(op_name, arg)
        typename = item["typename"]
        name = item["name"]
        attrs.append(item)
    return attrs


def parse_layout_transform(
    op_name: str, layout_transform: Dict[str, Any]
) -> Tuple[str, List]:
    if layout_transform is None:
        return "", []
    return layout_transform["arg_name"], parse_plain_list(
        layout_transform["tensors"]
    )
