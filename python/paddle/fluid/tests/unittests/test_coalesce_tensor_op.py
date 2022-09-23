#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
from paddle.fluid import core
import paddle.fluid as fluid
import paddle


def coalesce_tensor_eager_api(Input,
                              datatype=core.VarDesc.VarType.FP32,
                              copy_data=False,
                              set_constant=False,
                              persist_output=False,
                              constant=0.0,
                              use_align=True,
                              align_size=-1,
                              user_defined_size_of_dtype=-1,
                              concated_shapes=[],
                              concated_ranks=[]):
    if datatype == int(core.VarDesc.VarType.FP32):
        datatype = core.VarDesc.VarType.FP32
    return paddle._C_ops.coalesce_tensor(Input, datatype, copy_data,
                                         set_constant, persist_output, constant,
                                         use_align, align_size,
                                         user_defined_size_of_dtype,
                                         concated_shapes, concated_ranks)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestAllocContinuousSpace(OpTest):

    def setUp(self):
        self.python_api = coalesce_tensor_eager_api
        self.op_type = "coalesce_tensor"
        self.dtype, self.fluid_dtype = self.init_dtype()
        self.attrs = self.init_attr()
        self.Inputs = self.init_input()
        self.Outputs, self.FusedOutput = self.init_output(
            self.Inputs, self.attrs["set_constant"], self.attrs["constant"])
        self.inputs = {'Input': self.Inputs}
        self.outputs = {'Output': self.Outputs, 'FusedOutput': self.FusedOutput}

    def init_dtype(self):
        return np.float32, int(core.VarDesc.VarType.FP32)

    def init_input(self):
        inputs = []
        inputs.append(("x1", np.random.random([20, 3]).astype(self.dtype)))
        inputs.append(("x2", np.random.random([20]).astype(self.dtype)))
        inputs.append(("x3", np.random.random([1]).astype(self.dtype)))
        inputs.append(("x4", np.random.random([200, 30]).astype(self.dtype)))
        inputs.append(("x5", np.random.random([30]).astype(self.dtype)))
        inputs.append(("x6", np.random.random([1]).astype(self.dtype)))
        return inputs

    def init_attr(self):
        return {
            "copy_data": True,
            "set_constant": False,
            "constant": 0.0,
            "dtype": self.fluid_dtype
        }

    def init_output(self, input_list, set_constant, constant):
        inputs = []
        outputs = input_list
        # GpuMinChunkSize=256 bytes, FP32=4 bytes
        alignment = 256 / 4
        if 'user_defined_size_of_dtype' in self.attrs:
            alignment = 256 / self.attrs['user_defined_size_of_dtype']

        for input in input_list:
            length = len(input[1].flatten())
            aligned_len = (length + alignment) // alignment * alignment
            out = np.zeros(int(aligned_len))
            out[0:length] = input[1].flatten()
            inputs.append(out)

        coalesce_tensor_var = np.concatenate([input for input in inputs])
        if set_constant:
            coalesce_tensor_var = np.ones((len(coalesce_tensor_var))) * constant
            outputs = [(out[0],
                        np.ones(out[1].shape).astype(self.dtype) * constant)
                       for out in outputs]
        return outputs, coalesce_tensor_var

    def verify_output(self, place):
        with fluid.dygraph.base.guard(place=place):
            tensor_input = [
                fluid.dygraph.base.to_variable(value=data[1])
                for data in self.inputs["Input"]
            ]
            eager_outputs, eager_fused_output = coalesce_tensor_eager_api(
                tensor_input,
                datatype=self.attrs["dtype"],
                copy_data=self.attrs["copy_data"]
                if "copy_data" in self.attrs else False,
                set_constant=self.attrs["set_constant"]
                if "set_constant" in self.attrs else False,
                persist_output=False,
                constant=self.attrs["constant"]
                if "constant" in self.attrs else 0.0,
                use_align=True,
                align_size=-1,
                user_defined_size_of_dtype=self.
                attrs["user_defined_size_of_dtype"]
                if "user_defined_size_of_dtype" in self.attrs else -1,
                concated_shapes=[],
                concated_ranks=[])
            for idx, (expected, eager_output) in enumerate(
                    zip(self.outputs['Output'], eager_outputs)):
                np.testing.assert_allclose(expected[1],
                                           eager_output,
                                           atol=1e-5,
                                           err_msg=f'not equal {idx}')
            np.testing.assert_allclose(self.outputs['FusedOutput'],
                                       eager_fused_output,
                                       atol=1e-5,
                                       err_msg=f'not equal fusedoutput')

    def test_check_output(self):
        self.check_output_with_place(place=core.CUDAPlace(0),
                                     no_check_set=["FusedOutput"],
                                     atol=1e-5)
        self.verify_output(core.CUDAPlace(0))


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestAllocContinuousSpace2(TestAllocContinuousSpace):

    def init_attr(self):
        return {
            "copy_data": False,
            "set_constant": True,
            "constant": 0.5,
            "dtype": self.fluid_dtype,
            "user_defined_size_of_dtype": 2
        }

    def test_check_output(self):
        self.check_output_with_place(place=core.CUDAPlace(0),
                                     no_check_set=["FusedOutput"],
                                     atol=1e-5)
        self.verify_output(core.CUDAPlace(0))


if __name__ == '__main__':
    unittest.main()
