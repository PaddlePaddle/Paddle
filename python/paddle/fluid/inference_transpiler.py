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
import os
import shutil
from . import core


class InferenceTranspiler:
    def transpile(self, program, scope, place):
        '''
        Transpile the program to a inference program by fused batch normalization.
 
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
        1. insert elementwise_add op when bias == 0, and adjust its input and output.
        2. fuse the batch_norm's parameters to conv and elementwise_add operators.
        3. remove batch_norm ops and its variables which are not used in any other ops.
        4. remove unused variables.

        :param program: program to transpile 
        :type program: Program
        :param scope: inference scope 
        :type scope: Scope
        :param place: inference place 
        :type place: Place
        :return: program by fused batch normalization
        :rtype: Program
        '''
        self.scope = scope
        self.place = place
        self.block_desc = program.get_desc().block(0)
        i = 0
        while i < self.block_desc.op_size():
            current_op = self.block_desc.op(i)
            # TODO(luotao1): consider only conv2d now. fc would be delt later.
            if current_op.type() in ['conv2d']:
                next_op = self.block_desc.op(i + 1)
                # TODO(luotao1): consider only conv2d without bias now.
                # If conv2d with bias, the next_op.type is elementwise_add.
                if (next_op.type() == 'batch_norm'):
                    # insert bias op
                    bias_op = self._insert_bias_op(i + 1, current_op, next_op)
                    program.sync_with_cpp()
                    # fuse batch_norm
                    self._fuse_param(current_op, next_op, bias_op)
                    # remove batch_norm_op
                    self.block_desc.remove_op(i + 2, i + 3)
                    program.sync_with_cpp()
                    i = i + 1
            i = i + 1

        self._remove_unused_var()
        program.sync_with_cpp()

        return program

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
        bias_op = self.block_desc.insert_op(index)
        bias_op.set_type("elementwise_add")
        # The input of bias_op is current_op's output and Bias of bn_op
        # The output of bias_op is bn_op's output
        bias_op.set_input("X", current_op.output("Output"))
        bias_op.set_input("Y", bn_op.input("Bias"))
        bias_op.set_output("Out", bn_op.output("Y"))
        bias_op.set_attr('axis', 1)  # dim_start=1
        return bias_op

    def _fuse_param(self, current_op, bn_op, bias_op):
        '''
        fuse the batch_norm_op' parameters to current_op (conv or fc)
        
        :param current_op: current operator (conv or fc)
        :type current_op: Operator
        :param bn_op: batch norm operator
        :type bn_op: Operator
        :param bias_op: elementwise_add operator for adding bias
        :type bias_op: Operator
        '''

        def _load_tensor(param_name):
            return self.scope.find_var(param_name[0]).get_tensor()

        def _load_param(param_name):
            return np.array(_load_tensor(param_name))

        bias_bn = _load_param(bn_op.input("Bias"))  #Bias
        scale_bn = _load_param(bn_op.input("Scale"))  #Scale
        mean_bn = _load_param(bn_op.input("Mean"))  #Mean
        var_bn = _load_param(bn_op.input("Variance"))  #Variance

        # TODO(luotao1): consider only conv2d now. fc would be delt later.
        current_param = _load_param(current_op.input("Filter"))
        current_tensor = _load_tensor(current_op.input("Filter"))

        std_bn = np.float32(np.sqrt(np.add(var_bn, 1e-5)))
        tmp = np.float32(np.divide(scale_bn, std_bn))

        # add bias of batch_norm_op to conv2d
        bias = np.zeros(bias_bn.shape)
        bias = np.float32(
            np.add(np.multiply(np.subtract(bias, mean_bn), tmp), bias_bn))
        bias_tensor = _load_tensor(bias_op.input("Y"))
        bias_tensor.set(bias, self.place)

        # re-compute weight of conv2d
        tmp = tmp.reshape(tmp.shape[0], -1)
        dst_param = current_param.reshape((tmp.shape[0], -1))
        dst_param = np.float32(np.multiply(dst_param, tmp))
        dst_param = dst_param.reshape(current_param.shape)

        # set the updated parameters
        current_tensor.set(np.array(dst_param), self.place)

    def _remove_unused_var(self):
        '''
        remove unused varibles in program desc
        '''
        args = []
        for i in xrange(0, self.block_desc.op_size()):
            current_op = self.block_desc.op(i)
            args += current_op.input_arg_names()
            args += current_op.output_arg_names()
        args = list(set(args))  # unique the input and output arguments

        for var in self.block_desc.all_vars():
            if var.name() not in args:
                self.block_desc.remove_var(var.name())
