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
import paddle.fluid.core as core
from paddle.fluid.framework import Program
from paddle.fluid.executor import global_scope


class Float16Transpiler:
    def transpile(self, program, place, scope=None):
        '''
        Transpile the program desc and cast the weights to float16 data type to
        enable float16 inference.

        Since the operator in a program desc will automatically choose the
        right compute kernel to run based on the data type of the input tensor.
        We actually don't need to change the program desc to run in float16 mode.

        However, in this way, users who are used to feeding and fetching tensors 
        of float32 data type when running typical inference may find it confusing
        and difficult to run inference in float16 mode as they need to convert
        input data to float16 dtype and then convert the results back to float32 
        dtype to match the rest of code.

        So this function appends cast ops to the program desc where necessary so 
        that users are able to run inference in float16 mode while providing input 
        tensor (feed_holder) of float data type and obtaining output tensor 
        (fetch_holder) of float data type. 

        Moreover, it is desired that when we have the scope and program desc to run
        inference in float32 mode, we can use a single API to do the necessary 
        modification and then user can run float16 inference on the fly. To make 
        this happen, this function also create new parameters in the scope to have the 
        converted float16 weights and change the operators in program desc to use 
        these new parameters.

        :param program: program to transpile 
        :type program: Program
        :param place: inference place 
        :type place: Place
        :param scope: inference scope 
        :type scope: Scope         
        '''
        if not isinstance(program, Program):
            raise TypeError("program should be as Program type")
        if not isinstance(place, core.CPUPlace) and not isinstance(
                place, core.CUDAPlace):
            raise TypeError("place should be as CPUPlace/CUDAPlace type")
        if scope is None:
            scope = global_scope()
        if not isinstance(scope, core._Scope):
            raise TypeError("scope should be as Scope type or None")

        self.scope = scope
        self.place = place
        self.block = program.block(0)
        self.input_map = {}  # store the input names should be adjusted 

        self._modify_feed_fetch()
        self._convert_param_to_float16()
        self._adjust_input(skip=True)
        self._remove_unused_var()

        # TODO(luotao): use clone() method to flush the program.desc in force, 
        # since some large program.desc will not be flushed immediately. 
        # And a better solution will be considered later.
        program = program.clone()

    # ====================== private transpiler functions =====================
    def _adjust_input(self, skip=False):
        '''
        Change the input variable name in operators.

        When we are in the process of modifying a program desc, we usually 
        replace some variables with some other variables, where we create 
        a dictionary input_map to record the one-to-one correspondence
        between each old variable and the new one. 

        After that, this function will search all the operators that use the 
        old variables and change the info in op to use the new variables. There 
        maybe some exceptions to this rule when we are using the float16 transpiler
        and insert cast ops to cast float32 variable to float16 one. After we 
        insert the cast op to cast var_1 to var_1_fp16, we don't want to change 
        the input of cast op to var_1_fp16 after using this function.     
        '''
        skip_ops = {"cast"}
        for i in range(len(self.block.ops)):
            current_op = self.block.ops[i]
            if skip and current_op.type in skip_ops:
                continue
            for input_arg in current_op.input_arg_names:
                if input_arg in self.input_map:
                    current_op._rename_input(input_arg,
                                             self.input_map[input_arg])

    def _remove_unused_var(self):
        '''
        remove unused varibles in program
        '''
        args = []
        for i in range(len(self.block.ops)):
            current_op = self.block.ops[i]
            args += current_op.input_arg_names
            args += current_op.output_arg_names
        args = list(set(args))  # unique the input and output arguments

        for var in self.block.vars.keys():
            if var not in args:
                self.block._remove_var(var)

    def _modify_feed_fetch(self):
        '''
        Modify feed fetch op/vars for float16 inference.

        For each feed op:
        feed_op->feed_target_var
        
        Change it to:
        feed_op->feed_target_var->cast_op(from other dtype to float16)->tmp_var

        For each fetch op:
        fetch_target_var->fetch_op

        Change it to:
        tmp_var->cast_op(from float16 to other dtype)->fetch_target_var->fetch_op

        :return: None
        '''

        def find_op(var):
            # It is possible that var.op is not up to date after some 
            # modifications to program desc. Here we force to make it up to date.
            var.op = None
            for op in self.block.ops:
                if var.name in op.output_arg_names:
                    var.op = op
                    break

            if var.op is None:
                raise ValueError("The target variable must have an "
                                 "associated operator that generates it.")

        i = 0
        while i < len(self.block.ops):
            cur_op = self.block.ops[i]
            if cur_op.type == "feed":
                var_name = cur_op.output("Out")[0]
                tmp_var_name = var_name + ".fp16"
                var = self.block.vars[var_name]
                tmp_var = self.block.create_var(
                    name=tmp_var_name.encode('ascii'),
                    type=var.type,
                    dtype=core.VarDesc.VarType.FP16,
                    shape=var.shape,
                    persistable=var.persistable)
                self.block._insert_op(
                    i + 1,
                    type="cast",
                    inputs={"X": var},
                    outputs={"Out": tmp_var},
                    attrs={
                        'in_dtype': int(var.dtype),
                        'out_dtype': int(tmp_var.dtype)
                    })
                self.input_map[var_name] = tmp_var_name
                i = i + 1
            elif cur_op.type == "fetch":
                var_name = cur_op.input("X")[0]
                tmp_var_name = var_name + ".fp16"
                var = self.block.vars[var_name]
                tmp_var = self.block.create_var(
                    name=tmp_var_name.encode('ascii'),
                    type=var.type,
                    dtype=core.VarDesc.VarType.FP16,
                    shape=var.shape,
                    persistable=var.persistable)
                find_op(var)
                var.op._rename_output(var_name, tmp_var_name)
                self.block._insert_op(
                    i,
                    type="cast",
                    inputs={"X": tmp_var},
                    outputs={"Out": var},
                    attrs={
                        'in_dtype': int(tmp_var.dtype),
                        'out_dtype': int(var.dtype)
                    })
                i = i + 1
            i = i + 1

    def _convert_param_to_float16(self):
        def _get_no_fp16_conversion_var_names():
            '''
            Get the set of input variable names that shouldn't be converted to float16.

            When we want to run inference in float16 mode, most parameters need to be 
            firstly converted to float16. However, there are some parameters that 
            shouldn't be converted to float16 because the corresponding operator 
            requires float32 parameters even in float16 mode (when the input data is 
            of float16 data type). Currently, the only operator that has this exclusion 
            is the batch norm op.

            :return: set of input variable names 
            :type var_names: set         
            '''
            op_names = {'batch_norm'}
            var_names = []
            for op in self.block.ops:
                if op.type in op_names:
                    var_names += op.input_arg_names
            return set(var_names)

        def _should_be_converted(var):
            return var.persistable and \
                   var.name not in self.no_conversion_vars and \
                   var.type != core.VarDesc.VarType.FEED_MINIBATCH and \
                   var.type != core.VarDesc.VarType.FETCH_LIST

        self.no_conversion_vars = _get_no_fp16_conversion_var_names()
        conversion_var_list = filter(_should_be_converted,
                                     self.block.vars.values())
        for var in conversion_var_list:
            fp16_var_name = var.name + ".fp16"
            fp16_var = self.block.create_parameter(
                name=fp16_var_name.encode('ascii'),
                type=var.type,
                dtype=core.VarDesc.VarType.FP16,
                shape=var.shape)

            # cast the data in the tensor of the original var to float16
            # data type and store it in the tensor of the new float16 var
            self.scope.var(fp16_var_name)
            fp16_tensor = self.scope.find_var(fp16_var_name).get_tensor()
            tensor = np.array(self.scope.find_var(var.name).get_tensor())
            # After the old tensor data is converted to np.float16, view(np.uint16)
            # is used so that the internal memory of the numpy array will be 
            # reinterpreted to be of np.uint16 data type, which is binded to fluid 
            # float16 data type via the help of pybind in tensor_py.h. 
            fp16_tensor.set(
                tensor.astype(np.float16).view(np.uint16), self.place)

            # old var will be replaced by the fp16 var in program desc
            self.input_map[var.name] = fp16_var_name
            self.block._remove_var(var.name)
