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
from framework import Program
from executor import global_scope
from . import core


class InferenceTranspiler:
    def transpile(self, program, place, scope=None):
        '''
        Transpile the program. Support only fuse batch normalization now.

        :param program: program to transpile 
        :type program: Program
        :param place: inference place 
        :type place: Place
        :param scope: inference scope 
        :type scope: Scope or None
        '''
        if not isinstance(program, Program):
            raise TypeError("program should be as Program type")
        if not isinstance(place, core.CPUPlace) and not isinstance(
                place, core.CUDAPlace):
            raise TypeError("place should be as CPUPlace/CUDAPlace type")
        if scope is None:
            scope = global_scope()
        if not isinstance(scope, core.Scope):
            raise TypeError("scope should be as Scope type or None")
        self.fuse_batch_norm(program, place, scope)

    def fuse_batch_norm(self, program, place, scope):
        '''
        Transpile the program by fused batch normalization.
 
        The batch normalization followed the convolution or fully connected layer 
        can be integrated with them. Doing so will give us a forward acceleration, 
        especially in environments like mobile or embedded.
                    
        For input X:
        - Conv process:        X = input * W + bias 
        - Batch norm process:  X' = (X - mean) / std 
        - Scale Process:       Y = a * X' + b

        After fuse into one operation:

        Y = (input * W + bias - mean) / std * a + b
          = input * a * W / std + ((bias - mean) / std * a + b)

        The operator transformation is: 
        - before:
          - conv->batch_norm->any_other_op (bias == 0)
          - conv->elementwise_add->batch_norm->any_other_op (bias != 0)
        - after: 
          - conv->elementwise_add->any_other_op
        
        The transpile stages are:
        1. insert elementwise_add op when bias == 0.
        2. fuse the batch_norm's parameters to conv and elementwise_add operators.
        3. remove batch_norm ops which are not used in any other ops.
        4. adjust the input of any_other_op to be the output of elementwise_add operator.
        5. remove unused variables.

        :param program: program to transpile 
        :type program: Program
        :param place: inference place 
        :type place: Place
        :param scope: inference scope 
        :type scope: Scope
        '''
        self.scope = scope
        self.place = place
        self.block = program.block(0)
        self.input_map = {}  # store the input names should be adjusted 

        i = 0
        while i < len(self.block.ops):
            current_op = self.block.ops[i]
            # TODO(luotao1): consider only conv2d now. fc would be delt later.
            if current_op.type in ['conv2d']:
                # TODO(luotao1): consider single chain network now. 
                # For branch network, we counldn't use block.ops[i + 1] as 
                # the judgment condition.
                next_op = self.block.ops[i + 1]
                # conv2d without bias
                if (next_op.type == 'batch_norm'):
                    # insert bias op
                    bias_op = self._insert_bias_op(i + 1, current_op, next_op)
                    # fuse batch_norm
                    self._fuse_param(current_op, next_op, bias_op, 0)
                    # remove batch_norm_op
                    self.block.remove_op(i + 2)
                    i = i + 1
                # conv2d with bias, the next_op.type is elementwise_add
                elif (next_op.type == 'elementwise_add'):
                    next_next_op = self.block.ops[i + 2]
                    if (next_next_op.type == 'batch_norm'):
                        # fuse batch_norm
                        self._fuse_param(current_op, next_next_op, next_op, 1)
                        # remove batch_norm_op
                        self.block.remove_op(i + 2)
                        i = i + 1
            i = i + 1

        self._adjust_input()
        self._remove_unused_var()
        # TODO(luotao): use clone() method to flush the program.desc in force, 
        # since some large program.desc will not be flushed immediately. 
        # And a better solution will be considered later.
        program = program.clone()

    def convert_to_float16(self, program, place, scope=None):
        '''
        '''
        if scope is None:
            scope = global_scope()

        self.scope = scope
        self.place = place
        self.block = program.block(0)
        self.input_map = {}  # store the input names should be adjusted 

        self._modify_feed_fetch()
        self._convert_param_to_float16()
        self._adjust_input()
        self._remove_unused_var()
        # TODO(luotao): use clone() method to flush the program.desc in force, 
        # since some large program.desc will not be flushed immediately. 
        # And a better solution will be considered later.
        program = program.clone()

    # ====================== private transpiler functions =====================
    def _insert_bias_op(self, index, current_op, bn_op):
        '''
        Construct elementwise_add operator for adding bias 
        and insert it into program.
        
        :param index: insert location of bias_op
        :type index: Int
        :param current_op: current operator (conv or fc)
        :type current_op: Operator
        :param bn_op: batch norm operator
        :type bn_op: Operator
        :return: bias_op
        :rtype: Operator
        '''
        # The input of bias_op is current_op's output and Bias of bn_op
        # The output of bias_op is bn_op's output
        x_var = self.block.var(current_op.output("Output")[0])
        y_var = self.block.var(bn_op.input("Bias")[0])
        out_var = self.block.var(bn_op.output("Y")[0])

        bias_op = self.block.insert_op(
            index,
            type="elementwise_add",
            inputs={"X": x_var,
                    "Y": y_var},
            outputs={"Out": out_var},
            attrs={"axis": 1})  # dim_start=1
        return bias_op

    def _fuse_param(self, current_op, bn_op, bias_op, with_bias):
        '''
        fuse the batch_norm_op' parameters to current_op (conv or fc)
        
        :param current_op: current operator (conv or fc)
        :type current_op: Operator
        :param bn_op: batch norm operator
        :type bn_op: Operator
        :param bias_op: elementwise_add operator for adding bias
        :type bias_op: Operator
        :param with_bias: If current operator has bias, with_bias = 1; otherwise 0. 
        :type with_bias: Int
        '''

        def _update_param(op, old_param_name, new_param):
            # For the sake of remaining the original variables the same as before,
            # create new variables in scope to store the new parameters.
            old_param_name = old_param_name[0]
            old_var = self.block.vars[old_param_name]
            new_param_name = old_param_name + '_fuse_bn'
            new_var = self.block.create_parameter(
                name=new_param_name.encode('ascii'),
                type=old_var.type,
                dtype=old_var.dtype,
                shape=old_var.shape)
            op.rename_input(old_param_name, new_param_name)
            self.scope.var(new_param_name)

            tensor = self.scope.find_var(new_param_name).get_tensor()
            tensor.set(np.array(new_param), self.place)

        def _load_param(param_name):
            return np.array(self.scope.find_var(param_name[0]).get_tensor())

        bias_bn = _load_param(bn_op.input("Bias"))  #Bias
        scale_bn = _load_param(bn_op.input("Scale"))  #Scale
        mean_bn = _load_param(bn_op.input("Mean"))  #Mean
        var_bn = _load_param(bn_op.input("Variance"))  #Variance

        # TODO(luotao1): consider only conv2d now. fc would be delt later.
        current_param = _load_param(current_op.input("Filter"))
        std_bn = np.float32(np.sqrt(np.add(var_bn, 1e-5)))
        tmp = np.float32(np.divide(scale_bn, std_bn))

        # add bias of batch_norm_op to conv2d
        if with_bias:
            bias = _load_param(bias_op.input("Y"))
        else:
            bias = np.zeros(bias_bn.shape)
        bias = np.float32(
            np.add(np.multiply(np.subtract(bias, mean_bn), tmp), bias_bn))

        # re-compute weight of conv2d
        tmp = tmp.reshape(tmp.shape[0], -1)
        dst_param = current_param.reshape((tmp.shape[0], -1))
        dst_param = np.float32(np.multiply(dst_param, tmp))
        dst_param = dst_param.reshape(current_param.shape)

        # update parameters
        _update_param(current_op, current_op.input("Filter"), dst_param)
        _update_param(bias_op, bias_op.input("Y"), bias)

        # collect the renamed input
        self.input_map[bn_op.output("Y")[0]] = bias_op.output("Out")[0]

    def _adjust_input(self):
        for i in range(len(self.block.ops)):
            current_op = self.block.ops[i]
            for input_arg in current_op.input_arg_names:
                if input_arg in self.input_map:
                    current_op.rename_input(input_arg,
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
                self.block.remove_var(var)

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
            for op in self.block.ops:
                if var.name in op.output_arg_names:
                    var.op = op
                    break

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
                    shape=var.shape)
                self.block.insert_op(
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
                    shape=var.shape)
                find_op(var)
                var.op.rename_output(var_name, tmp_var_name)
                self.block.insert_op(
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

        no_conversion_vars = _get_no_fp16_conversion_var_names()
        for var_name, var in self.block.vars.iteritems():
            if var.persistable and var_name not in no_conversion_vars and \
                   var.type != core.VarDesc.VarType.FEED_MINIBATCH and \
                   var.type != core.VarDesc.VarType.FETCH_LIST:
                fp16_var_name = var_name + ".fp16"
                self.input_map[var_name] = fp16_var_name
                fp16_var = self.block.create_parameter(
                    name=fp16_var_name.encode('ascii'),
                    type=var.type,
                    dtype=core.VarDesc.VarType.FP16,
                    shape=var.shape)

                self.scope.var(fp16_var_name)
                fp16_tensor = self.scope.find_var(fp16_var_name).get_tensor()
                tensor = np.array(self.scope.find_var(var_name).get_tensor())
                fp16_tensor.set(
                    tensor.astype(np.float16).view(np.uint16), self.place)

                # old var will be replaced by the fp16 var in program desc
                self.input_map[var_name] = fp16_var_name
                self.block.remove_var(var_name)
