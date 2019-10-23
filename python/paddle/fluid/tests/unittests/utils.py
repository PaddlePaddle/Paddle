# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.framework import _dygraph_guard
import paddle.fluid as fluid
from paddle.fluid.framework import Variable
import paddle.fluid.dygraph.jit as jit
from paddle.fluid.dygraph.jit import extract_vars
import numpy as np
import os
import time

__all__ = ['DyGraphProgramDescTracerTestHelper', ]


def is_equal_program(prog1, prog2):
    with _dygraph_guard(None):
        return _is_equal_program(prog1, prog2)


def _is_equal_program(prog1, prog2):
    block_num = prog1.num_blocks
    if block_num != prog2.num_blocks:
        return False

    for block_id in range(block_num):
        block1 = prog1.block(block_id)
        block2 = prog2.block(block_id)

        if len(block1.ops) != len(block2.ops):
            return False

        if len(block1.vars) != len(block2.vars):
            return False

        for op1, op2 in zip(block1.ops, block2.ops):
            if op1.input_arg_names != op2.input_arg_names:
                return False

            if op1.output_arg_names != op2.output_arg_names:
                return False

            attr1 = op1.all_attrs()
            attr2 = op2.all_attrs()

            if len(attr1) != len(attr2):
                return False

            for key1, value1 in attr1.items():
                if key1 not in attr2:
                    return False

                if value1 != attr2.get(key1):
                    return False

        for var1 in block1.vars.values():
            if var1.name not in block2.vars:
                return False

            var2 = block2.vars.get(var1.name)
            if var1.name != var2.name:
                return False

            if var1.type != var2.type:
                return False

            if var1.dtype != var2.dtype:
                return False

            if var1.lod_level != var2.lod_level:
                return False

            if var1.persistable != var2.persistable:
                return False

    return True


def load_dygraph_vars_to_scope(model_path, scope, place):
    def load_dict_to_scope(scope, dictionary):
        if scope is None:
            scope = fluid.global_scope()

        for k, v in dictionary.items():
            dst_t = scope.var(k).get_tensor()
            src_t = v.value().get_tensor()
            dst_t.set(np.array(src_t), place)
            dst_t.set_lod(src_t.lod())

    param_dict, opti_dict = fluid.load_dygraph(model_path)
    if param_dict:
        load_dict_to_scope(scope, param_dict)

    if opti_dict:
        load_dict_to_scope(scope, opti_dict)


class DyGraphProgramDescTracerTestHelper(object):
    def __init__(self,
                 module,
                 unittest_obj,
                 model_path=None,
                 scope=None,
                 place=None):
        self.module = module
        self.unittest_obj = unittest_obj
        self.scope = fluid.Scope() if scope is None else scope

        self.model_path = model_path
        if model_path is None:
            millis = int(round(time.time() * 1000))
            self.model_path = "id_{}_{}".format(id(module), millis)

        self.place = place
        if place is None:
            self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()

        self.program = None

        self.executor = fluid.Executor(self.place)

    def _remove_model_path(self):
        if os.path.exists(self.model_path + ".pdparams"):
            os.remove(self.model_path + ".pdparams")

        if os.path.exists(self.model_path + ".pdopt"):
            os.remove(self.model_path + ".pdopt")

    def _run_static_graph(self, inputs, feed_names, fetch_names):
        var_list = extract_vars(inputs)
        assert len(var_list) == len(feed_names)

        feed_dict = {}
        for name, var in zip(feed_names, var_list):
            feed_dict[name] = np.array(var.value().get_tensor())

        with fluid.scope_guard(self.scope):
            with _dygraph_guard(None):
                return self.executor.run(self.program,
                                         feed=feed_dict,
                                         fetch_list=fetch_names)

    def run(self, inputs, feed_names, fetch_names):
        out_dygraph, program = jit.trace(
            self.module, inputs, feed_names=feed_names, fetch_names=fetch_names)

        if self.program is not None:
            self.unittest_obj.assertTrue(
                is_equal_program(self.program, program))

        self.program = program

        fluid.save_dygraph(self.module.state_dict(), self.model_path)
        load_dygraph_vars_to_scope(self.model_path, self.scope, self.place)

        self._remove_model_path()

        out_static_graph = self._run_static_graph(inputs, feed_names,
                                                  fetch_names)

        if not isinstance(out_dygraph, (list, tuple)):
            assert len(out_static_graph) == 1
            out_static_graph = out_static_graph[0]

        return out_dygraph, out_static_graph

    def assertEachVar(self, out_dygraph, out_static_graph, func=None):
        if func is None:
            func = lambda x, y: np.array_equal(x, y)

        if not isinstance(out_dygraph, (list, tuple)):
            out_dygraph = [out_dygraph]

        if not isinstance(out_static_graph, (list, tuple)):
            out_static_graph = [out_static_graph]

        for v1, v2 in zip(out_dygraph, out_static_graph):
            self.unittest_obj.assertTrue(func(v1.numpy(), v2))
