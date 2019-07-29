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

from __future__ import print_function

import os
import unittest
import numpy as np
import random
import six
import time
import itertools
import collections
from collections import defaultdict
from op_test import OpTest

import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.backward import append_backward
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, OpProtoHolder, Variable
from testsuite import create_op, set_input, append_input_output, append_loss_ops


class OpInplaceTest(OpTest):
    def _calc_output(self,
                     place,
                     enable_inplace=False,
                     parallel=False,
                     no_check_set=None,
                     loss=None):
        program = Program()
        block = program.global_block()
        self._append_ops(block)

        inputs = self._get_inputs(block)
        outputs = self._get_outputs(block)
        feed_map = self.feed_var(inputs, place)

        if parallel:
            use_cuda = False
            if isinstance(place, fluid.CUDAPlace(0)):
                use_cuda = True
            if loss:
                executor = fluid.ParallelExecutor(
                    use_cuda=use_cuda,
                    loss_name=loss.name,
                    main_program=program)
            else:
                executor = fluid.ParallelExecutor(
                    use_cuda=use_cuda, main_program=program)
        else:
            executor = Executor(place)

        fetch_list = getattr(self, "fetch_list", [])
        # if the fetch_list is customized by user, we use it directly.
        # if not, fill the fetch_list by the user configured outputs in test.
        if len(fetch_list) == 0:
            for var_name, var in six.iteritems(outputs):
                if no_check_set is not None and var_name in no_check_set:
                    continue
                if isinstance(var, list):
                    for v in var:
                        fetch_list.append(v)
                else:
                    fetch_list.append(var)
        # if the fetch_list still empty, fill the fetch_list by the operator output.
        if len(fetch_list) == 0:
            for out_name, out_dup in Operator.get_op_outputs(self.op_type):
                fetch_list.append(str(out_name))
        # fetch_list = map(block.var, fetch_list)
        if not isinstance(fetch_list[0], fluid.framework.Variable):
            fetch_list = list(map(block.var, fetch_list))

        build_strategy = fluid.BuildStrategy()
        build_strategy.memory_optimize = self.memory_optimize
        build_strategy.enable_inplace = enable_inplace
        build_strategy.fuse_all_optimizer_ops = self.fuse_all_optimizer_ops

        compiled_prog = fluid.CompiledProgram(program).with_data_parallel(
            build_strategy=build_strategy, places=place)

        outs = executor.run(compiled_prog,
                            feed=feed_map,
                            fetch_list=fetch_list,
                            return_numpy=False)
        return outs, fetch_list

    def check_output_with_place(self,
                                place,
                                atol=0,
                                no_check_set=None,
                                equal_nan=False,
                                check_dygraph=False):
        actual_outs, fetch_list = self._calc_output(place, enable_inplace=True)
        expect_outs, fetch_list = self._calc_output(place, enable_inplace=False)

        for i, var in enumerate(fetch_list):
            if isinstance(expect_outs[i], tuple):
                print("expect_outs is tuple!")
                self.assertTrue(
                    np.array_equal(
                        np.array(expect_outs[i][0]),
                        np.array(actual_outs[i][0])))
            else:
                self.assertTrue(
                    np.array_equal(
                        np.array(expect_outs[i]), np.array(actual_outs[i])))

    def check_output(self,
                     atol=1e-5,
                     no_check_set=None,
                     equal_nan=False,
                     check_dygraph=False):
        places = self._get_places()
        for place in places:
            self.check_output_with_place(place, atol, no_check_set, equal_nan,
                                         check_dygraph)
