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

import os
import unittest
import warnings
import numpy as np
import random
import six
import struct
import time
import itertools
import collections
from collections import defaultdict

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.backward import append_backward
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, OpProtoHolder, Variable, convert_np_dtype_to_dtype_
from testsuite import create_op, set_input, append_input_output, append_loss_ops
from paddle.fluid import unique_name
from white_list import op_accuracy_white_list, check_shape_white_list, compile_vs_runtime_white_list, no_check_set_white_list
from white_list import op_threshold_white_list, no_grad_set_white_list
from op_test import OpTest, _set_use_system_allocator, get_numeric_gradient
from xpu.get_test_cover_info import is_empty_grad_op_type, get_xpu_op_support_types, type_dict_str_to_numpy


class XPUOpTest(OpTest):

    @classmethod
    def setUpClass(cls):
        '''Fix random seeds to remove randomness from tests'''
        cls.use_xpu = True
        cls.use_mkldnn = False
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        """Restore random seeds"""

        def is_empty_grad_op(op_type):
            grad_op = op_type + '_grad'
            xpu_version = core.get_xpu_device_version(0)
            xpu_op_list = core.get_xpu_device_op_list(xpu_version)
            if grad_op in xpu_op_list.keys():
                return False
            return True

        if cls.dtype == np.float16:
            place = paddle.XPUPlace(0)
            if core.is_float16_supported(place) == False:
                return

        if cls.dtype == np.float64:
            return

        super().tearDownClass()

    def _get_places(self):
        places = [paddle.XPUPlace(0)]
        return places

    def check_output(self,
                     atol=0.001,
                     no_check_set=None,
                     equal_nan=False,
                     check_dygraph=True,
                     inplace_atol=None,
                     check_eager=False):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place, atol, no_check_set, equal_nan,
                                     check_dygraph, inplace_atol, check_eager)

    def check_output_with_place(self,
                                place,
                                atol=0.001,
                                no_check_set=None,
                                equal_nan=False,
                                check_dygraph=True,
                                inplace_atol=None,
                                check_eager=False):
        self.infer_dtype_from_inputs_outputs(self.inputs, self.outputs)
        if self.dtype == np.float64:
            return

        if self.dtype == np.float16:
            if core.is_float16_supported(place) == False:
                return

        if self.dtype == np.float16:
            atol = 0.1
        return super().check_output_with_place(place, atol, no_check_set,
                                               equal_nan, check_dygraph,
                                               inplace_atol)

    def check_grad(self,
                   inputs_to_check,
                   output_names,
                   no_grad_set=None,
                   numeric_grad_delta=0.005,
                   in_place=False,
                   max_relative_error=0.005,
                   user_defined_grads=None,
                   user_defined_grad_outputs=None,
                   check_dygraph=True,
                   numeric_place=None,
                   check_eager=False):
        place = paddle.XPUPlace(0)
        self.check_grad_with_place(place, inputs_to_check, output_names,
                                   no_grad_set, numeric_grad_delta, in_place,
                                   max_relative_error, user_defined_grads,
                                   user_defined_grad_outputs, check_dygraph,
                                   numeric_place, check_eager)

    def check_grad_with_place(self,
                              place,
                              inputs_to_check,
                              output_names,
                              no_grad_set=None,
                              numeric_grad_delta=0.005,
                              in_place=False,
                              max_relative_error=0.005,
                              user_defined_grads=None,
                              user_defined_grad_outputs=None,
                              check_dygraph=True,
                              numeric_place=None,
                              check_eager=False):
        if hasattr(self, 'op_type_need_check_grad'):
            xpu_version = core.get_xpu_device_version(0)
            if is_empty_grad_op_type(xpu_version, self.op_type,
                                     self.in_type_str):
                self._check_grad_helper()
                return

        cast_grad_op_types = get_xpu_op_support_types('cast')
        cast_grad_op_types_np = []
        for ctype in cast_grad_op_types:
            cast_grad_op_types_np.append(type_dict_str_to_numpy[ctype])

        if (self.dtype not in cast_grad_op_types_np):
            return

        if self.dtype == np.float64:
            return

        if self.dtype == np.float16:
            if core.is_float16_supported(place) == False:
                return

        if self.dtype == np.float16:
            max_relative_error = 1.0
            return super().check_grad_with_place(
                place, inputs_to_check, output_names, no_grad_set,
                numeric_grad_delta, in_place, max_relative_error,
                user_defined_grads, user_defined_grad_outputs, check_dygraph)

        a1 = self.get_grad_with_place(
            place,
            inputs_to_check,
            output_names,
            no_grad_set=no_grad_set,
            user_defined_grad_outputs=user_defined_grad_outputs)
        a2 = self.get_grad_with_place(
            place,
            inputs_to_check,
            output_names,
            no_grad_set=no_grad_set,
            user_defined_grad_outputs=user_defined_grad_outputs)
        a3 = self.get_grad_with_place(
            paddle.CPUPlace(),
            inputs_to_check,
            output_names,
            no_grad_set=no_grad_set,
            user_defined_grad_outputs=user_defined_grad_outputs)
        self._assert_is_close(a1, a2, inputs_to_check, 0.00000001,
                              "Gradient Check On two xpu")
        self._assert_is_close(a1, a3, inputs_to_check, max_relative_error,
                              "Gradient Check On cpu & xpu")

    def get_grad_with_place(self,
                            place,
                            inputs_to_check,
                            output_names,
                            no_grad_set=None,
                            numeric_grad_delta=0.005,
                            in_place=False,
                            max_relative_error=0.005,
                            user_defined_grad_outputs=None,
                            check_dygraph=True):
        self.scope = core.Scope()
        op_inputs = self.inputs if hasattr(self, "inputs") else dict()
        op_outputs = self.outputs if hasattr(self, "outputs") else dict()
        op_attrs = self.attrs if hasattr(self, "attrs") else dict()

        self._check_grad_helper()
        if self.dtype == np.float64 and \
            self.op_type not in op_threshold_white_list.NEED_FIX_FP64_CHECK_GRAD_THRESHOLD_OP_LIST:
            numeric_grad_delta = 1e-5
            max_relative_error = 1e-7

        cache_list = None
        if hasattr(self, "cache_name_list"):
            cache_list = self.cache_name_list

        # oneDNN numeric gradient should use CPU kernel
        use_onednn = False
        if "use_mkldnn" in op_attrs and op_attrs["use_mkldnn"] == True:
            op_attrs["use_mkldnn"] = False
            use_onednn = True

        mean_grad_op_types = get_xpu_op_support_types('mean')
        mean_grad_op_types_np = []
        for mtype in mean_grad_op_types:
            mean_grad_op_types_np.append(type_dict_str_to_numpy[mtype])

        self.op = create_op(self.scope,
                            self.op_type,
                            op_inputs,
                            op_outputs,
                            op_attrs,
                            cache_list=cache_list)

        if use_onednn:
            op_attrs["use_mkldnn"] = True

        if no_grad_set is None:
            no_grad_set = set()
        else:
            if (self.op_type not in no_grad_set_white_list.NEED_TO_FIX_OP_LIST
                ) and (self.op_type
                       not in no_grad_set_white_list.NOT_CHECK_OP_LIST) and (
                           not self.is_bfloat16_op()):
                raise AssertionError("no_grad_set must be None, op_type is " +
                                     self.op_type + " Op.")

        for input_to_check in inputs_to_check:
            set_input(self.scope, self.op, self.inputs, place)

        if not type(output_names) is list:
            output_names = [output_names]

        if (self.dtype not in mean_grad_op_types_np):

            prog = Program()
            block = prog.global_block()
            scope = core.Scope()
            self._append_ops(block)

            inputs = self._get_inputs(block)
            outputs = self._get_outputs(block)
            feed_dict = self.feed_var(inputs, place)
            cast_inputs = list(map(block.var, output_names))
            cast_outputs = block.create_var(dtype="float32",
                                            shape=cast_inputs[0].shape)
            cast_op = block.append_op(type="cast",
                                      inputs={"X": cast_inputs},
                                      outputs={"Out": cast_outputs},
                                      attrs={
                                          "in_dtype":
                                          convert_np_dtype_to_dtype_(
                                              self.dtype),
                                          "out_dtype":
                                          core.VarDesc.VarType.FP32
                                      })
            cast_op.desc.infer_var_type(block.desc)
            cast_op.desc.infer_shape(block.desc)

            output_names = [cast_outputs.name]

            loss = append_loss_ops(block, output_names)
            loss_names = [loss.name]
            recast_inputs = list(map(block.var, loss_names))
            recast_loss = block.create_var(dtype=self.dtype,
                                           shape=recast_inputs[0].shape)

            recast_op = block.append_op(type="cast",
                                        inputs={"X": recast_inputs},
                                        outputs={"Out": recast_loss},
                                        attrs={
                                            "in_dtype":
                                            core.VarDesc.VarType.FP32,
                                            "out_dtype":
                                            convert_np_dtype_to_dtype_(
                                                self.dtype)
                                        })
            recast_op.desc.infer_var_type(block.desc)
            recast_op.desc.infer_shape(block.desc)

            param_grad_list = append_backward(loss=recast_loss,
                                              parameter_list=[input_to_check],
                                              no_grad_set=no_grad_set)
            fetch_list = [g for p, g in param_grad_list]

            executor = fluid.Executor(place)
            return list(
                map(
                    np.array,
                    executor.run(prog,
                                 feed_dict,
                                 fetch_list,
                                 scope=scope,
                                 return_numpy=False)))

        analytic_grads = self._get_gradient(
            inputs_to_check,
            place,
            output_names,
            no_grad_set,
            user_defined_grad_outputs=user_defined_grad_outputs)
        return analytic_grads
