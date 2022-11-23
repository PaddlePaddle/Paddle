#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np

import paddle.fluid.core as core
import paddle.fluid as fluid


class TestElementWiseAddOp(unittest.TestCase):

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        np.testing.assert_allclose(np.array(tensor),
                                   np_array,
                                   rtol=1e-05,
                                   atol=atol,
                                   err_msg=msg)

    def check_forward_backward(self):

        def test_with_place(place):
            out_grad = np.random.random_sample(self.x.shape).astype(np.float32)
            x_grad = out_grad
            sum_axis = list(range(0, len(self.x.shape)))
            del sum_axis[self.axis]
            y_grad = np.sum(out_grad, axis=tuple(sum_axis))

            var_dict = locals()
            var_dict['y'] = self.y
            var_dict['x'] = self.x
            var_dict['out'] = self.out
            var_dict['y@GRAD'] = y_grad
            var_dict['x@GRAD'] = x_grad
            var_dict['out@GRAD'] = out_grad

            var_names = ['x', 'y', 'out', 'y@GRAD', 'x@GRAD', 'out@GRAD']
            ground_truth = {name: var_dict[name] for name in var_names}

            program = fluid.Program()
            with fluid.program_guard(program):
                block = program.global_block()
                for name in ground_truth:
                    block.create_var(name=name,
                                     dtype='float32',
                                     shape=ground_truth[name].shape)
                elementwise_add_op = block.append_op(type="elementwise_add",
                                                     inputs={
                                                         "X": block.var('x'),
                                                         "Y": block.var('y'),
                                                     },
                                                     outputs={
                                                         "Out":
                                                         block.var('out'),
                                                     },
                                                     attrs={
                                                         "axis": self.axis,
                                                     })

                # generate backward op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    elementwise_add_op.desc, set(), [])
                grad_op_desc = grad_op_desc_list[0]
                new_op_desc = block.desc.append_op()
                new_op_desc.copy_from(grad_op_desc)
                for var_name in grad_op_desc.output_arg_names():
                    block.desc.var(var_name.encode("ascii"))
                grad_op_desc.infer_var_type(block.desc)
                grad_op_desc.infer_shape(block.desc)
                for arg in grad_op_desc.output_arg_names():
                    grad_var = block.desc.find_var(arg.encode("ascii"))
                    grad_var.set_dtype(core.VarDesc.VarType.FP32)

                exe = fluid.Executor(place)
                out = exe.run(program,
                              feed={
                                  name: var_dict[name]
                                  for name in ['x', 'y', 'out@GRAD']
                              },
                              fetch_list=['x@GRAD', 'y@GRAD'])
                self.__assert_close(x_grad, out[0], "x@GRAD")
                self.__assert_close(y_grad, out[1], "y@GRAD", atol=1.4)

        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
                "elementwise_add"):
            places.append(core.CUDAPlace(0))

        for place in places:
            test_with_place(place)

    def test_check_forward_backward_with_scale_and_bias(self):
        np.random.seed(123)
        self.x = np.random.random((4, 32, 220, 220)).astype(np.float32)
        self.y = np.random.random((32)).astype(np.float32)
        self.out = self.x + self.y.reshape(1, 32, 1, 1)
        self.axis = 1
        self.check_forward_backward()


if __name__ == '__main__':
    unittest.main()
