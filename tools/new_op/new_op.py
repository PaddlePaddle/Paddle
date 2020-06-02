# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import re

import template
from template import *


def _underscores_to_camel(name):
    """
    Convert name form underscores case to camel case. For example, 'under_scores' to 'UnderScores'.
    Args:
        name(str): The original name, which is underscores case.
    Returns:
        name(str): The name of camel case.
    """
    res = ''
    for item in name.split('_'):
        res += (item[0].upper() + item[1:])
    return res


def _get_op_class_name(op_name):
    return _underscores2camel(op_name) + "Op"


def _get_grad_op_class_name(op_name):
    return _underscores2camel(op_name) + "GradOp"


def _get_grad_op_name(op_name):
    return op_name + '_grad'


def _number_to_order(x):
    """
    Convert a number to order str. For example, '1' to '1st'.
    """
    if x <= 0:
        raise ValueError("The number should be larger than 0")
    if (x // 10) % 10 == 1:
        return str(x) + 'th'
    elif x % 10 == 1:
        return str(x) + 'st'
    elif x % 10 == 2:
        return str(x) + 'nd'
    elif x % 10 == 3:
        return str(x) + 'rd'
    else:
        return str(x) + 'th'


class NewOpGenerator(object):
    def __init__(self,
                 op_name,
                 input_names,
                 output_names,
                 attr_names,
                 output_dir=None):
        if not output_dir:
            default_output_dir = os.path.dirname(os.path.abspath(
                __file__)) + '/../../paddle/fluid/operators'
            self.output_dir = default_output_dir
        else:
            self.output_dir = output_dir
        self.op_name = op_name
        self.input_names = input_names
        self.output_names = output_names
        self.attr_names = attr_names

        self.op_class_name = _get_op_class_name(op_name)
        self.grad_op_name = _get_grad_op_name(op_name)
        self.grad_op_class_name = _get_grad_op_class_name(op_name)

    def generate_header_file(self):
        header_file = self.output_dir + '/' + self.op_name + '_op.h'
        with open(header_file, 'w') as f:
            f.write(template.copyright_template)
            f.write(template.header_template)

    def generate_cc_file(self):
        cc_file = self.output_dir + '/' + self.op_name + '_op.cc'
        with open(cc_file, 'w') as f:
            f.write(template.copyright_template)
            f.write(
                Template(op_class_template).format(
                    op_class_name=self.op_class_name))

    def generate_cuda_file(self):
        pass

    def generate_op_files(self):
        self.generate_header_file()
        self.generate_cc_file()
        self.generate_cuda_file()


if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1].isdigit():
    #     check_approval(int(sys.argv[1]), sys.argv[2:])
    # else:
    #     print(
    #         "Usage: python check_pr_approval.py [count] [required reviewer id] ..."
    #     )
    print(_number_to_order(1), _number_to_order(2), _number_to_order(3),
          _number_to_order(4), _number_to_order(11), _number_to_order(121))
    gen = NewOpGenerator("mul2", ["X", "Y"], ["Out"], ["attr1", "attr2"])
    print(gen.output_dir)
    print(gen.op_class_name)
    gen.generate_op_files()
    print(os.path.dirname(os.path.abspath(__file__)))
    print(template.op_class_template)
