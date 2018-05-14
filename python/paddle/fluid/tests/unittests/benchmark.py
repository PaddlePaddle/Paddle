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

import numpy as np
import unittest
import random
import time

import paddle.fluid as fluid
# from paddle.fluid.backward import append_backward
# from paddle.fluid.op import Operator
# from paddle.fluid.executor import Executor
# from paddle.fluid.framework import Program, OpProtoHolder
# from testsuite import create_op, set_input, append_input_output, as_lodtensor, append_loss_ops
from op_test import OpTest


class BenchmarkSuite(OpTest):
    def timeit_function(self, callback, iters, *args):
        assert iters != 0, "Iters should >= 1"
        start = time.time()
        for i in range(iters):
            callback(place, *args)
            elapse = time.time() - start
        return elapse / iters

    def timeit_output(self, iters=100, place):
        return self.timeit_function(self.calc_output, iters, place)

    def timeit_grad(self, place, iters=100):
        pass


class BenchmarkSuite2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''Fix random seeds to remove randomness from tests'''
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()

        np.random.seed(123)
        random.seed(123)

    @classmethod
    def tearDownClass(cls):
        '''Restore random seeds'''
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

    def feed_var(self, input_vars, place):
        feed_map = {}
        for var_name in input_vars:
            if isinstance(input_vars[var_name], list):
                for name, np_value in self.inputs[var_name]:
                    tensor = core.LoDTensor()
                    if isinstance(np_value, tuple):
                        tensor.set(np_value[0], place)
                        tensor.set_lod(np_value[1])
                    else:
                        tensor.set(np_value, place)
                    feed_map[name] = tensor
            else:
                tensor = core.LoDTensor()
                if isinstance(self.inputs[var_name], tuple):
                    tensor.set(self.inputs[var_name][0], place)
                    tensor.set_lod(self.inputs[var_name][1])
                else:
                    tensor.set(self.inputs[var_name], place)
                feed_map[var_name] = tensor

        return feed_map

    def _get_io_vars(self, block, numpy_inputs):
        inputs = {}
        for name, value in numpy_inputs.iteritems():
            if isinstance(value, list):
                var_list = [
                    block.var(sub_name) for sub_name, sub_value in value
                ]
                inputs[name] = var_list
            else:
                inputs[name] = block.var(name)
        return inputs

    def _get_inputs(self, block):
        return self._get_io_vars(block, self.inputs)

    def _get_outputs(self, block):
        return self._get_io_vars(block, self.outputs)

    def _append_ops(self, block):
        op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)
        inputs = append_input_output(block, op_proto, self.inputs, True)
        outputs = append_input_output(block, op_proto, self.outputs, False)
        op = block.append_op(
            type=self.op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=self.attrs if hasattr(self, "attrs") else dict())
        # infer variable type and infer shape in compile-time
        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)

    def get_output_with_place(self, place, parallel=False):
        program = Program()
        block = program.global_block()
        self._append_ops(block)
        loss = append_loss_ops(block, output_names)
        param_grad_list = append_backward(
            loss=loss, parameter_list=input_to_check, no_grad_set=no_grad_set)
        feed_dict = self.feed_var(inputs, place)

        fetch_list = [g for p, g in param_grad_list]
        if parallel:
            use_cuda = False
            if isinstance(place, fluid.CUDAPlace(0)):
                use_cuda = True
            executor = fluid.ParallelExecutor(
                use_cuda=use_cuda, loss_name=loss.name, main_program=program)
        else:
            executor = Executor(place)
        outs = executor.run(program, feed_dict, fetch_list, return_numpy=True)

    def assert_allclose(self, outs, fetch_list, atol):
        pass

    def check_output(self, atol):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(self.op_type):
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_output_with_place(place, atol)
