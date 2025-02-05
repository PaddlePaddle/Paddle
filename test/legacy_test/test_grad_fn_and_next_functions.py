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
"""
Test the tensor attribute grad_fn and the properties of the reverse node grad_node, such as next_function
"""

import unittest

import paddle
from paddle import nn


class Testmodel(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x**2
        y = x + y
        return y


class TestAnonmousSurvey(unittest.TestCase):
    """
    Test the tensor attribute grad_fn and the properties of the reverse node grad_node, such as next_function

    """

    def init_graph(self):
        """define reversed graph

        func_name [str]: represents the name of the operator node
        next_funcs [dict]: represents the operator node
        """
        self.grad_fn_1 = {"func_name": "GradNodeAccumulation", "next_funcs": {}}
        self.grad_fn_2 = {
            "func_name": "PowGradNode",
            "next_funcs": {"GradNodeAccumulation": self.grad_fn_1},
        }
        self.grad_fn_3 = {
            "func_name": "AddGradNode",
            "next_funcs": {
                "GradNodeAccumulation": self.grad_fn_1,
                "PowGradNode": self.grad_fn_2,
            },
        }
        self.output_grad_fn = {"grad_fn": self.grad_fn_3}

    def init_data(self):
        """define output of model

        the final output will be saved self.output
        """
        model = Testmodel()
        x = paddle.randn([1, 3, 24, 24])
        x.stop_gradient = False
        self.output = model(x)

    def setUp(self):
        self.init_graph()
        self.init_data()

    def test_grad_fn_and_next_funs(self):
        self.check_func(self.output.grad_fn, self.output_grad_fn["grad_fn"])

    def check_func(self, grad_fn, grad_fn_json) -> None:
        """
        Check each node, grad_fn is tensor attribute. grad_fn_json is structure of next_node.

        Args:
            grad_fn (grad_fn): grad_fn of node
            grad_fn_json (dict): grad_node_json of node
        """
        self.assertEqual(grad_fn.name(), grad_fn_json["func_name"])
        # Recursively test other nodes
        if hasattr(grad_fn, 'next_functions') and grad_fn.next_functions[0]:
            next_funcs_json = grad_fn_json["next_funcs"]
            for u in grad_fn.next_functions:
                self.check_func(u, next_funcs_json[u.name()])


if __name__ == "__main__":
    unittest.main()
