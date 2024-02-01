#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

import logging
import os
import sys

from cinn.common import is_compiled_with_cuda
from cinn.framework import Scope
from cinn.frontend import PaddleModelConvertor

import paddle
from paddle.base.layer_helper import LayerHelper
from paddle.static import Variable as PaddleVariable

sys.path.append("/work/dev_CINN/build/python/tests")
from test.cinn.ops.op_test import OpTest

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(name="op_test")

paddle.enable_static()


class OpMapperTest(OpTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_place()
        self.init_input_data()

    def _init_place(self):
        self.place = paddle.CPUPlace()
        if is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

    def init_input_data(self):
        self.feed_data = {}
        logger.warn("No Input Data")

    def set_op_type(self) -> str:
        """Set paddle C++ op type:\n
        The op type should be got from the paddle static program.
        Not the paddle python api name or phi api name.\n
        For example, the C++ op type of `paddle.sum` is `reduce_sum`, the code from `Paddle/python/paddle/tensor/math.py`:
        ```
        def sum(x, axis=None, dtype=None, keepdim=False, name=None):
            ...
             helper.append_op(
                type='reduce_sum',
                inputs={'X': x},
                outputs={'Out': out},
                attrs=attrs,
            )
        ```
        """
        raise Exception("Not implemented.")

    def set_op_inputs(self) -> dict:
        """Map from input parameter name to argument list, the argument should be get from paddle.static.data.\n
        For example, `concat` should return
        ```
        x1 = paddle.static.data(name='x1', shape=[1, 2], dtype='float32')
        x2 = paddle.static.data(name='x2', shape=[1, 2], dtype='float32')
        return {'X' : [x1, x2]}
        ```"""
        return {}

    def set_op_attrs(self) -> dict:
        """Map from attribute name to attribute value:\n
        For example, `concat` should return
        ```
        return {'axis' : 0}
        ```
        """
        return {}

    def set_op_outputs(self) -> dict:
        """Map from output parameter name to argument type, the argument type should be represented by a string.\n
        For example, if the `out_dtype` attribute of `cast` is `'float16'`, here should return
        ```
        return {'Out' : 'float16'}
        ```
        """
        raise Exception("Not implemented.")

    def skip_check_outputs(self) -> set:
        """Skip check some output because some paddle's op outputs are useless, CINN will not support these.
        ```
        # skip check the result of output 'Out'
        return {'Out'}
        ```
        """
        return set()

    def set_inplace_outputs(self) -> dict:
        """Map from inplace output parameter name to input parameter name.\n
        For example, if the op's output 'MeanOut' should share the memory with the input 'Mean', here should return
        ```
        return {'MeanOut' : 'Mean'}
        ```
        """
        return {}

    def __set_paddle_op(self):
        # paddle C++ op type
        self.op_type = self.set_op_type()
        # map from input param name to argument name list
        self.inputs = self.set_op_inputs()
        # map from attribute name to attribute value
        self.attrs = self.set_op_attrs()
        # map from output param name to output data type
        self.output_dtypes = self.set_op_outputs()
        # list of outputs which will be skip
        self.skip_outputs = self.skip_check_outputs()
        # dict of inplace var
        self.inplace_outputs = self.set_inplace_outputs()
        # collect some important infomation
        self.input_arg_map = self.__get_arguments_map(self.inputs)
        self.fetch_targets = []
        self.skip_check_list = []
        self.op_desc = None

    def __check_valid(self):
        self.assertIsInstance(
            self.op_type, str, msg="The op type should be a string"
        )
        self.assertNotEqual(
            self.op_type, "", msg="The op type should not empty"
        )
        self.assertIsInstance(
            self.inputs,
            dict,
            msg="The set_op_inputs should be return dict(InputName, list(Variable)), where Variable are created by paddle.static.data",
        )
        self.assertIsInstance(
            self.attrs,
            dict,
            msg="The set_op_attrs should be return dict(AttrName, AttrValue)",
        )
        self.assertIsInstance(
            self.output_dtypes,
            dict,
            msg="The set_op_outputs should be return dict(OutName, list(OutDtype)), where OutName and OutDtype are string",
        )
        self.assertGreater(
            len(self.output_dtypes),
            0,
            msg="The set_op_outputs cannot return a empty dict",
        )

        for name, var in self.input_arg_map.items():
            self.assertIn(name, self.feed_data)
            self.assertEqual(
                var.shape,
                self.feed_data[name].shape,
                msg=f"The shape of input {var.name} in feed_data is error",
            )
            self.assertEqual(
                self.paddleddtype2nptype(var.dtype),
                str(self.feed_data[name].dtype),
                msg=f"The dtype of input {var.name} in feed_data is error",
            )

        for out_name, in_name in self.inplace_outputs.items():
            self.assertNotIn(
                out_name,
                self.output_dtypes,
                msg="The {} should not declare twice because it's a inplace output, you should remove it from \"set_op_outputs\"".format(
                    out_name
                ),
            )
            self.assertIn(
                in_name,
                self.inputs,
                msg="The inplace var should existed in op' inputs dict",
            )

    def __get_arguments_map(self, param_maps):
        arg_maps = {}
        for args in param_maps.values():
            self.assertIsInstance(
                args,
                list,
                msg="The type of arguments should be list(Variable), where Variable are created by paddle.static.data",
            )
            for var in args:
                self.assertIsInstance(
                    var,
                    PaddleVariable,
                    msg="The type of argument should be paddle.static.Variable",
                )
                self.assertTrue(
                    (var.name not in arg_maps) or (arg_maps[var.name] == var),
                    msg="Argument %s is duplicated" % var.name,
                )
                arg_maps[var.name] = var
        return arg_maps

    def __init_paddle_op(self):
        self.__set_paddle_op()
        self.__check_valid()

    def __remove_skip_outputs(self, results):
        check_outputs = []
        for i in range(len(self.fetch_targets)):
            if self.fetch_targets[i].name not in self.skip_check_list:
                check_outputs.append(results[i])
                logger.debug(
                    msg="{}, shape={}, dtype={}:\n{}".format(
                        self.fetch_targets[i].name,
                        results[i].shape,
                        str(results[i].dtype),
                        results[i],
                    )
                )

        return check_outputs

    def __debug_numpy_dict(self, info_dict: dict, title: str):
        if logger.isEnabledFor(logging.DEBUG):
            debug_info = ""
            for k, v in info_dict.items():
                debug_info += (
                    k
                    + ", shape="
                    + str(v.shape)
                    + ", dtype="
                    + str(v.dtype)
                    + ":\n"
                )
                debug_info += str(v) + "\n"

            logger.debug(title + ":\n" + debug_info)

    def build_paddle_program(self, target):
        self.__debug_numpy_dict(self.feed_data, "Feed Data")

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            self.__init_paddle_op()
            helper = LayerHelper(self.op_type)

            self.outputs = {}
            for var_name, dtypes in self.output_dtypes.items():
                self.assertIsInstance(
                    dtypes,
                    list,
                    msg="The set_op_outputs should be return dict(OutName, list(OutDtype)), where OutName and OutDtype are string",
                )
                self.outputs[var_name] = []
                for dtype in dtypes:
                    out_var = helper.create_variable_for_type_inference(dtype)
                    self.fetch_targets.append(out_var)
                    self.outputs[var_name].append(out_var)
                    if var_name in self.skip_outputs:
                        self.skip_check_list.append(out_var.name)

            # inplace output
            for out_name, in_name in self.inplace_outputs.items():
                self.outputs[out_name] = self.inputs[in_name]
                for var in self.inputs[in_name]:
                    self.fetch_targets.append(var)
                    if out_name in self.skip_outputs:
                        self.skip_check_list.append(var.name)

            self.op_desc = helper.append_op(
                type=self.op_type,
                inputs=self.inputs,
                outputs=self.outputs,
                attrs=self.attrs,
            ).desc

        logger.debug("Paddle Program:\n" + str(main_program))

        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)

        results = exe.run(
            main_program,
            self.feed_data,
            fetch_list=self.fetch_targets,
            return_numpy=True,
        )

        # NOTE: The unittest of `test_reduce_op`, `test_argmax_op`, `test_argmin_op` will
        # output 0D-Tensor, hence we need to reshape them into 1D-Tensor temporarily.
        # After corresponding CINN op supports 0D-Tensor, this trick can be removed safely.
        for i in range(len(results)):
            if results[i] is not None and len(results[i].shape) == 0:
                results[i] = results[i].reshape(1)

        logger.debug("Paddle result:")
        self.paddle_outputs = self.__remove_skip_outputs(results)

    def build_cinn_program(self, target):
        scope = Scope()
        convertor = PaddleModelConvertor(target=self.target, scope=scope)

        for var_name, var in self.input_arg_map.items():
            convertor.create_input(
                dtype=self.paddleddtype2nptype(var.dtype),
                shape=var.shape,
                name=var_name,
            )

        convertor.append_op(
            type=self.op_type,
            inputs=self.op_desc.inputs(),
            outputs=self.op_desc.outputs(),
            attrs=self.attrs,
        )

        prog = convertor()

        logger.debug("CINN Program:\n" + str(prog))

        # get the CINN input list
        cinn_inputs = []
        cinn_feed_datas = []

        vars = self.get_program_vars(prog)

        # map the name the variable
        if len(self.input_arg_map) > 0:
            feed_names = set(self.input_arg_map.keys())
            for name in feed_names:
                cinn_name = convertor.get_cinn_name(name)

                self.assertIn(
                    cinn_name,
                    vars,
                    msg="Cannot find variable "
                    + cinn_name
                    + " in cinn program's var list",
                )
                cinn_inputs.append(vars[cinn_name])
                cinn_feed_datas.append(self.feed_data[name])

        # get the CINN output list
        fetch_names = []
        inplace_start = 0
        for dtypes in self.output_dtypes.values():
            inplace_start += len(dtypes)
        fetch_names += [var.name for var in self.fetch_targets[:inplace_start]]

        inplace_end = inplace_start
        for in_name in self.inplace_outputs.values():
            inplace_end += len(self.inputs[in_name])
        fetch_names += [
            var.name + "@InplaceOut"
            for var in self.fetch_targets[inplace_start:inplace_end]
        ]

        # map the name the variable
        self.assertGreater(
            len(fetch_names), 0, msg="The program's output cannot be empty!"
        )
        cinn_output_vars = []
        for name in fetch_names:
            cinn_name = convertor.get_cinn_name(name)

            self.assertIn(
                cinn_name,
                vars,
                msg="Cannot find variable "
                + cinn_name
                + " in cinn program's var list",
            )
            cinn_output_vars.append(vars[cinn_name])

        # run and get result
        results = self.get_cinn_output(
            prog,
            target,
            cinn_inputs,
            cinn_feed_datas,
            cinn_output_vars,
            passes=[],
            scope=scope,
        )

        logger.debug("CINN result:")
        self.cinn_outputs = self.__remove_skip_outputs(results)

    @staticmethod
    def get_program_vars(program) -> dict:
        vars = {}
        for i in range(program.size()):
            instr = program[i]
            for var in instr.get_inputs():
                if var.id() not in vars:
                    vars[var.id()] = var
            for var in instr.get_outputs():
                if var.id() not in vars:
                    vars[var.id()] = var

        return vars

    @staticmethod
    def paddleddtype2nptype(dtype):
        switch_map = {
            paddle.float16: "float16",
            paddle.float32: "float32",
            paddle.float64: "float64",
            paddle.int8: "int8",
            paddle.int16: "int16",
            paddle.int32: "int32",
            paddle.int64: "int64",
            paddle.uint8: "uint8",
            paddle.bool: "bool",
            paddle.base.core.VarDesc.VarType.RAW: "unk",
        }
        assert dtype in switch_map, str(dtype) + " not support in CINN"
        return switch_map[dtype]

    @staticmethod
    def nptype2paddledtype(dtype):
        switch_map = {
            "float16": paddle.float16,
            "float32": paddle.float32,
            "float64": paddle.float64,
            "int8": paddle.int8,
            "int16": paddle.int16,
            "int32": paddle.int32,
            "int64": paddle.int64,
            "uint8": paddle.uint8,
            "bool": paddle.bool,
            # The paddle's phi::DataType::UNDEFINED is mapped into ProtoDataType::RAW,
            "unk": paddle.base.core.VarDesc.VarType.RAW,
        }
        assert dtype in switch_map, dtype + " not support in CINN"
        return switch_map[dtype]
