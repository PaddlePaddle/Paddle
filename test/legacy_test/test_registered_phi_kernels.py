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

import pathlib
import re
import unittest

import yaml

from paddle.base import core


def parse_kernels_name(op_item):
    result = []
    if 'kernel' in op_item:
        kernel_config = op_item['kernel']
        kernel_funcs = re.compile(r'([a-zA-Z0-9_]+)\s*({[^}]+})?').findall(
            kernel_config['func']
        )
        for func_item in kernel_funcs:
            result.append(func_item[0])

    return result


def get_all_kernels(op_list, all_registered_kernels):
    kernels = []
    for op in op_list:
        op_kernels = parse_kernels_name(op)
        for op_kernel in op_kernels:
            if op_kernel not in kernels and op_kernel in all_registered_kernels:
                kernels.append(op_kernel)
            if op_kernel not in all_registered_kernels:
                print("********** wrong kernel: ", op_kernel)
    return kernels


def remove_forward_kernels(bw_kernels, forward_kernels):
    new_bw_kernels = []
    for bw_kernel in bw_kernels:
        if bw_kernel not in forward_kernels:
            new_bw_kernels.append(bw_kernel)
    return new_bw_kernels


class TestRegisteredPhiKernels(unittest.TestCase):
    """TestRegisteredPhiKernels."""

    def setUp(self):
        self.forward_ops = []
        self.backward_ops = []

        root_path = pathlib.Path(__file__).parents[3]

        ops_yaml_path = [
            'paddle/phi/ops/yaml/ops.yaml',
            'paddle/phi/ops/yaml/inconsistent/dygraph_ops.yaml',
        ]
        bw_ops_yaml_path = [
            'paddle/phi/ops/yaml/backward.yaml',
            'paddle/phi/ops/yaml/inconsistent/dygraph_backward.yaml',
        ]

        for each_ops_yaml in ops_yaml_path:
            with open(root_path.joinpath(each_ops_yaml), 'r') as f:
                op_list = yaml.load(f, Loader=yaml.FullLoader)
                if op_list:
                    self.forward_ops.extend(op_list)

        for each_ops_yaml in bw_ops_yaml_path:
            with open(root_path.joinpath(each_ops_yaml), 'r') as f:
                api_list = yaml.load(f, Loader=yaml.FullLoader)
                if api_list:
                    self.backward_ops.extend(api_list)

    def test_registered_phi_kernels(self):
        phi_function_kernel_infos = core._get_registered_phi_kernels("function")
        registered_kernel_list = list(phi_function_kernel_infos.keys())
        forward_kernels = get_all_kernels(
            self.forward_ops, registered_kernel_list
        )
        backward_kernels = remove_forward_kernels(
            get_all_kernels(self.backward_ops, registered_kernel_list),
            forward_kernels,
        )

        for kernel_name in forward_kernels:
            self.assertIn(kernel_name, registered_kernel_list)

        for kernel_name in backward_kernels:
            self.assertIn(kernel_name, registered_kernel_list)


if __name__ == '__main__':
    unittest.main()
