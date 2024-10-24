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

import argparse
import re


class TileDesc:
    def __init__(self, Tshape, stages, Wshape, math_inst):
        self.Tshape = Tshape
        self.stages = stages
        self.Wshape = Wshape
        self.math_inst = math_inst


def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = f"\\$\\{{{key}\\}}"
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


def parse_args():
    parser = argparse.ArgumentParser(
        description="The argument for generating the conv2d_bias_act kernels."
    )

    parser.add_argument(
        "--cuda_arch",
        type=str,
        default=None,
        help="The CUDA architecture to be generated.",
    )
    args = parser.parse_args()

    return args


def write_kernel_to_file(kernel, file_name):
    with open(
        file_name,
        "w",
    ) as f:
        f.write(kernel)
        f.close()
