# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class InputOutputTensorsExtractor:

    def __init__(self, block_func):
        self.block_func = block_func
        self.input_tensors = []
        self.output_tensors = []

    def Extract(self, free_vars, args):
        self.input_tensors += list(free_vars)
        self.input_tensors += list(args)
        self.block_func(self, *free_vars)(*args)
        return self.input_tensors, self.output_tensors

    def pd_op_data(self, op):
        self.input_tensors += list(op.GetResults())

    def pd_op_feed(self, op):
        self.input_tensors += list(op.GetResults())

    def builtin_parameter(self, op):
        self.input_tensors += list(op.GetResults())

    def builtin_constant(self, op):
        self.input_tensors += list(op.GetResults())

    def builtin_shadow_output(self, op, *inputs):
        self.output_tensors += list(inputs)

    def pd_op_fetch(self, op, *inputs):
        self.output_tensors += list(inputs)

    def cf_yield(self, op, *inputs):
        self.output_tensors += list(inputs)

    def __call__(self, op, *input_tensors, **kwargs):
        method_name = op.GetPyVarName()
        if hasattr(self, method_name):
            getattr(self, method_name)(op, *input_tensors)
        return op.GetResults()
