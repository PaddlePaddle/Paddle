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

import paddle

all_dir = list(
    set(
        dir(paddle.Tensor)
        + dir(paddle.base.framework.Variable)
        + dir(paddle.base.libpaddle.pir.OpResult)
    )
)
all_dir.sort()
output_text = """|         math          |                        动态图Tensor                         |                      老IR Varialbe                      |                      新IR OpResult                      |
|:---------------------:|:--------------------------------------------------------:|:------------------------------------------------------:|:------------------------------------------------------:|
"""


print(len(all_dir))
for i in all_dir:
    math_name = f"`{i}`" if "__" in i else i
    temp_line = f"| {math_name} "
    if i in dir(paddle.Tensor):
        temp_line += "| ✅ "
    else:
        temp_line += "| ❌ "

    if i in dir(paddle.base.framework.Variable):
        temp_line += "| ✅ "
    else:
        temp_line += "| ❌ "

    if i in dir(paddle.base.libpaddle.pir.OpResult):
        # api 对应 pr 匹配规则
        if i in paddle.tensor.tensor_method_func:
            temp_line += "| #57857 |"
        else:
            match i:
                case "place" | "ndimension" | "dim" | "ndim" | "item":
                    temp_line += "| #58042 |"
                case "astype":
                    temp_line += "| #58026 |"
                case "__div__" | "__truediv__" | "__rdiv__" | "__rtruediv__":
                    temp_line += "| #57997 |"
                case "__rmul__" | "__mul__" | "__rsub__" | "__sub__" | "__radd__" | "__add__":
                    temp_line += "| #58106 |"
                case "__pow__" | "__rpow__" | "__floordiv__" | "__mod__" | "__matmul__":
                    temp_line += "| #58219 |"
                case "__ne__" | "__lt__" | "__le__" | "__gt__" | "__ge__" | "__and__" | "__or__" | "__xor__" | "__invert__":
                    temp_line += "| #58343 |"
                case "clone":
                    temp_line += "| #59115 |"
                case "append":
                    temp_line += "| #59220 |"
                case "cpu" | "cuda":
                    temp_line += "| #59300 |"
                case "__neg__":
                    temp_line += "| #60166 |"
                case "__eq__":
                    temp_line += "| #58896 |"
                case _:
                    temp_line += "| ✅ |"
    else:
        temp_line += "| ❌ |"

    output_text += temp_line + "\n"

with open("./method_compare.md", "w") as file:
    file.write(output_text)
