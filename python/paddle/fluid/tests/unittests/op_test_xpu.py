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
from paddle.fluid.framework import Program, OpProtoHolder, Variable
from testsuite import create_op, set_input, append_input_output, append_loss_ops
from paddle.fluid import unique_name
from white_list import op_accuracy_white_list, check_shape_white_list, compile_vs_runtime_white_list, no_check_set_white_list
from white_list import op_threshold_white_list, no_grad_set_white_list
from op_test import OpTest, _set_use_system_allocator, get_numeric_gradient


class XPUOpTest(OpTest):
    @classmethod
    def setUpClass(cls):
        '''Fix random seeds to remove randomness from tests'''
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()
        cls.call_once = False
        cls.dtype = np.float32
        cls.outputs = {}
        cls.input_shape_is_large = True

        np.random.seed(123)
        random.seed(124)

        cls._use_system_allocator = _set_use_system_allocator(True)

    @classmethod
    def tearDownClass(cls):
        """Restore random seeds"""
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

        _set_use_system_allocator(cls._use_system_allocator)

        def is_empty_grad_op(op_type):
            all_op_kernels = core._get_all_register_op_kernels()
            grad_op = op_type + '_grad'
            if grad_op in all_op_kernels.keys():
                if is_mkldnn_op_test():
                    grad_op_kernels = all_op_kernels[grad_op]
                    for grad_op_kernel in grad_op_kernels:
                        if 'MKLDNN' in grad_op_kernel:
                            return False
                else:
                    return False
            return True

        def is_xpu_op_test():
            return True

        def is_mkldnn_op_test():
            return False

        if not hasattr(cls, "op_type"):
            raise AssertionError(
                "This test do not have op_type in class attrs, "
                "please set self.__class__.op_type=the_real_op_type manually.")

        # case in NO_FP64_CHECK_GRAD_CASES and op in NO_FP64_CHECK_GRAD_OP_LIST should be fixed
        if not hasattr(cls, "no_need_check_grad") \
            and not is_empty_grad_op(cls.op_type):
            if cls.dtype is not None and \
                cls.dtype != np.float32:
                raise AssertionError("This test of %s op needs check_grad." %
                                     cls.op_type)

    def try_call_once(self, data_type):
        if not self.call_once:
            self.call_once = True
            if data_type is not None and \
                data_type != np.float32:
                raise AssertionError("Unsupport data type %s in xpu" %
                                     data_type)
            self.dtype = data_type

    def check_output_with_place(self,
                                place,
                                atol=0.001,
                                no_check_set=None,
                                equal_nan=False,
                                check_dygraph=True,
                                inplace_atol=None):
        self.infer_dtype_from_inputs_outputs(self.inputs, self.outputs)
        if self.dtype == np.float64 and \
            self.op_type not in op_threshold_white_list.NEED_FIX_FP64_CHECK_OUTPUT_THRESHOLD_OP_LIST:
            atol = 0

        if self.is_bfloat16_op():
            check_dygraph = False
            if hasattr(self, 'force_fp32_output') and getattr(
                    self, 'force_fp32_output'):
                atol = 1e-2
            else:
                atol = 2

        if no_check_set is not None:
            if self.op_type not in no_check_set_white_list.no_check_set_white_list:
                raise AssertionError(
                    "no_check_set of op %s must be set to None." % self.op_type)

        if check_dygraph:
            dygraph_outs = self._calc_dygraph_output(
                place, no_check_set=no_check_set)
        outs, fetch_list = self._calc_output(place, no_check_set=no_check_set)
        for out_name, out_dup in Operator.get_op_outputs(self.op_type):
            if out_name not in self.outputs:
                continue
            if no_check_set is not None and out_name in no_check_set:
                continue

            def find_imperative_actual(target_name, dygraph_outs, place):
                with fluid.dygraph.base.guard(place=place):
                    for name in dygraph_outs:
                        if name == target_name:
                            return dygraph_outs[name][0]
                        var_list = dygraph_outs[name]
                        for i, var in enumerate(var_list):
                            if var.name == target_name:
                                return dygraph_outs[name][i]
                    self.assertTrue(False, "Found failed {} {}".format(
                        dygraph_outs.keys(), target_name))

            def find_actual(target_name, fetch_list):
                found = [
                    i for i, var_name in enumerate(fetch_list)
                    if var_name == target_name
                ]
                self.assertTrue(
                    len(found) == 1, "Found {} {}".format(
                        len(found), target_name))
                return found[0]

            if out_dup:
                sub_out = self.outputs[out_name]
                if not isinstance(sub_out, list):
                    raise AssertionError("sub_out type %s is not list",
                                         type(sub_out))
                for item in sub_out:
                    sub_out_name, expect = item[0], item[1]
                    if check_dygraph:
                        imperative_actual = find_imperative_actual(
                            sub_out_name, dygraph_outs, place)
                        imperative_actual_t = np.array(imperative_actual.value()
                                                       .get_tensor())
                    idx = find_actual(sub_out_name, fetch_list)
                    actual = outs[idx]
                    actual_t = np.array(actual)
                    expect_t = expect[0] \
                        if isinstance(expect, tuple) else expect
                    self.assertTrue(
                        np.allclose(
                            actual_t, expect_t, atol=atol, equal_nan=equal_nan),
                        "Output (" + sub_out_name + ") has diff at " +
                        str(place))
                    if check_dygraph:
                        self.assertTrue(
                            np.allclose(
                                imperative_actual_t,
                                expect_t,
                                atol=atol,
                                equal_nan=equal_nan),
                            "Output (" + sub_out_name + ") has diff at " +
                            str(place) + " in dygraph mode")
                    if isinstance(expect, tuple):
                        self.assertListEqual(
                            actual.recursive_sequence_lengths(), expect[1],
                            "Output (" + sub_out_name +
                            ") has different lod at " + str(place))
                        if check_dygraph:
                            self.assertListEqual(
                                imperative_actual.value().get_tensor()
                                .recursive_sequence_lengths(), expect[1],
                                "Output (" + out_name +
                                ") has different lod at " + str(place) +
                                " in dygraph mode")
            else:
                if check_dygraph:
                    imperative_actual = find_imperative_actual(
                        out_name, dygraph_outs, place)
                    imperative_actual_t = np.array(imperative_actual.value()
                                                   .get_tensor())
                idx = find_actual(out_name, fetch_list)
                actual = outs[idx]
                actual_t = np.array(actual)
                expect = self.outputs[out_name]
                expect_t = expect[0] if isinstance(expect, tuple) else expect
                self.assertTrue(
                    np.allclose(
                        actual_t, expect_t, atol=atol, equal_nan=equal_nan),
                    "Output (" + out_name + ") has diff at " + str(place) +
                    "\nExpect " + str(expect_t) + "\n" + "But Got" +
                    str(actual_t) + " in class " + self.__class__.__name__ + " "
                    + str(atol) + " " + str(expect_t - actual_t))
                if check_dygraph:
                    if six.moves.reduce(
                            lambda x, y: x * y, imperative_actual_t.shape,
                            1) == 0 and six.moves.reduce(
                                lambda x, y: x * y, expect_t.shape, 1) == 0:
                        pass
                    else:
                        self.assertTrue(
                            np.allclose(
                                imperative_actual_t,
                                expect_t,
                                atol=atol,
                                equal_nan=equal_nan),
                            "Output (" + out_name + ") has diff at " +
                            str(place) + "\nExpect " + str(expect_t) + "\n" +
                            "But Got" + str(imperative_actual_t) + " in class "
                            + self.__class__.__name__)
                if isinstance(expect, tuple):
                    self.assertListEqual(actual.recursive_sequence_lengths(),
                                         expect[1], "Output (" + out_name +
                                         ") has different lod at " + str(place))
                    if check_dygraph:
                        self.assertListEqual(
                            imperative_actual.value().get_tensor()
                            .recursive_sequence_lengths(), expect[1],
                            "Output (" + out_name + ") has different lod at " +
                            str(place) + " in dygraph mode")

        # Note(zhiqiu): inplace_atol should be only set when op doesn't ensure
        # computational consistency.
        # For example, group_norm uses AtomicAdd on CUDAPlace, which do not ensure
        # computation order when multiple threads write the same address. So the
        # result of group_norm is non-deterministic when datatype is float.
        # When inplace_atol is not None, the inplace check uses numpy.allclose
        # to check inplace result instead of numpy.array_equal.
        if inplace_atol is not None:
            warnings.warn(
                "inplace_atol should only be set when op doesn't ensure computational consistency, please check it!"
            )
        # Check inplace for given op, its grad op, its grad_grad op, etc.
        # No effect on original OpTest
        # Currently not support ParallelExecutor on XPUPlace.
        if not paddle.is_compiled_with_xpu():
            self.check_inplace_output_with_place(
                place, no_check_set=no_check_set, inplace_atol=inplace_atol)

        if check_dygraph:
            return outs
        else:
            return outs

    def check_grad_with_place(self,
                              place,
                              inputs_to_check,
                              output_names,
                              no_grad_set=None,
                              numeric_grad_delta=0.005,
                              in_place=False,
                              max_relative_error=0.005,
                              user_defined_grads=None,
                              check_dygraph=True):
        place = paddle.XPUPlace(0)
        a1 = self.get_grad_with_place(
            place, inputs_to_check, output_names, no_grad_set=no_grad_set)
        a2 = self.get_grad_with_place(
            place, inputs_to_check, output_names, no_grad_set=no_grad_set)
        a3 = self.get_grad_with_place(
            paddle.CPUPlace(),
            inputs_to_check,
            output_names,
            no_grad_set=no_grad_set)
        self._assert_is_close(a1, a2, inputs_to_check, 0.00000001,
                              "Gradient Check On two xpu")
        self._assert_is_close(a1, a3, inputs_to_check, 0.001,
                              "Gradient Check On cpu & xpu")

    def get_grad_with_place(self,
                            place,
                            inputs_to_check,
                            output_names,
                            no_grad_set=None,
                            numeric_grad_delta=0.005,
                            in_place=False,
                            max_relative_error=0.005,
                            user_defined_grads=None,
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

        self.op = create_op(
            self.scope,
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
                ) and (
                    self.op_type not in no_grad_set_white_list.NOT_CHECK_OP_LIST
                ) and (not self.is_bfloat16_op()):
                raise AssertionError("no_grad_set must be None, op_type is " +
                                     self.op_type + " Op.")

        for input_to_check in inputs_to_check:
            set_input(self.scope, self.op, self.inputs, place)
            tensor_to_check = self.scope.find_var(input_to_check).get_tensor()
            tensor_size = six.moves.reduce(lambda a, b: a * b,
                                           tensor_to_check.shape(), 1)
            if tensor_size < 100:
                self.__class__.input_shape_is_large = False

        if not type(output_names) is list:
            output_names = [output_names]

        analytic_grads = self._get_gradient(inputs_to_check, place,
                                            output_names, no_grad_set)
        return analytic_grads
